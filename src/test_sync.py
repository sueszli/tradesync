import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl

from sync import _optimize, _optimize_bank_transactions
from validate import validate_optimization


def calculate_net_balances(df: pl.DataFrame) -> dict:
    if len(df) == 0:
        return {}

    balances = {}
    for row in df.iter_rows(named=True):
        prod, acc_from, acc_to, quantity = row["ProductId"], row["AccFrom"], row["AccTo"], row["Quantity"]
        key_from = (prod, acc_from)
        key_to = (prod, acc_to)

        balances[key_from] = balances.get(key_from, 0) - quantity
        balances[key_to] = balances.get(key_to, 0) + quantity

    return {k: v for k, v in balances.items() if abs(v) > 1e-10}


def assert_balances_equal(balance1: dict, balance2: dict, tolerance: float = 1e-8):
    all_keys = set(balance1.keys()) | set(balance2.keys())

    for key in all_keys:
        val1 = balance1.get(key, 0)
        val2 = balance2.get(key, 0)
        assert abs(val1 - val2) <= tolerance, f"Balance mismatch for {key}: {val1} vs {val2}"


class TestOptimize:
    def test_empty_dataframe(self):
        empty_df = pl.DataFrame(
            {"ProductId": [], "AccFrom": [], "AccTo": [], "Quantity": []},
            schema={"ProductId": pl.Utf8, "AccFrom": pl.Utf8, "AccTo": pl.Utf8, "Quantity": pl.Float64},
        )
        result = _optimize(empty_df)
        assert len(result) == 0
        assert result.columns == ["ProductId", "AccFrom", "AccTo", "Quantity"]

    def test_single_transaction(self):
        df = pl.DataFrame({"ProductId": ["PROD1"], "AccFrom": ["ACC1"], "AccTo": ["ACC2"], "Quantity": [100.0]})
        result = _optimize(df)
        assert len(result) == 1
        assert result.row(0) == ("PROD1", "ACC1", "ACC2", 100.0)

    def test_bidirectional_netting(self):
        df = pl.DataFrame(
            {"ProductId": ["PROD1", "PROD1"], "AccFrom": ["ACC1", "ACC2"], "AccTo": ["ACC2", "ACC1"], "Quantity": [150.0, 50.0]}
        )
        result = _optimize(df)
        assert len(result) == 1
        assert result.row(0) == ("PROD1", "ACC1", "ACC2", 100.0)

    def test_complete_cancellation(self):
        df = pl.DataFrame(
            {"ProductId": ["PROD1", "PROD1"], "AccFrom": ["ACC1", "ACC2"], "AccTo": ["ACC2", "ACC1"], "Quantity": [100.0, 100.0]}
        )
        result = _optimize(df)
        assert len(result) == 0

    def test_multiple_parallel_flows(self):
        df = pl.DataFrame(
            {
                "ProductId": ["PROD1", "PROD1", "PROD1"],
                "AccFrom": ["ACC1", "ACC1", "ACC1"],
                "AccTo": ["ACC2", "ACC2", "ACC2"],
                "Quantity": [50.0, 75.0, 25.0],
            }
        )
        result = _optimize(df)
        assert len(result) == 1
        assert result.row(0) == ("PROD1", "ACC1", "ACC2", 150.0)

    def test_complex_three_account_scenario(self):
        df = pl.DataFrame(
            {
                "ProductId": ["PROD1"] * 6,
                "AccFrom": ["ACC1", "ACC2", "ACC3", "ACC2", "ACC3", "ACC1"],
                "AccTo": ["ACC2", "ACC3", "ACC1", "ACC1", "ACC2", "ACC3"],
                "Quantity": [100.0, 80.0, 60.0, 40.0, 30.0, 20.0],
            }
        )
        result = _optimize(df)
        original_balance = calculate_net_balances(df)
        result_balance = calculate_net_balances(result)
        assert_balances_equal(original_balance, result_balance)

    def test_self_transfers_ignored(self):
        df = pl.DataFrame(
            {"ProductId": ["PROD1", "PROD1"], "AccFrom": ["ACC1", "ACC1"], "AccTo": ["ACC1", "ACC2"], "Quantity": [50.0, 100.0]}
        )
        result = _optimize(df)
        # self-transfer should be ignored, only ACC1->ACC2 remains
        assert len(result) == 1
        assert result.row(0) == ("PROD1", "ACC1", "ACC2", 100.0)

    def test_floating_point_precision(self):
        df = pl.DataFrame(
            {
                "ProductId": ["PROD1", "PROD1"],
                "AccFrom": ["ACC1", "ACC2"],
                "AccTo": ["ACC2", "ACC1"],
                "Quantity": [100.000000001, 100.0],
            }
        )
        result = _optimize(df)
        # should detect this as effectively zero net flow
        # if not cancelled out completely, should be very small
        if len(result) > 0:
            assert abs(result["Quantity"][0]) < 1e-6


class TestOptimizeBankTransactions:
    def test_multiple_products_isolated(self):
        df = pl.DataFrame(
            {
                "ProductId": ["PROD1", "PROD1", "PROD2", "PROD2"],
                "AccFrom": ["ACC1", "ACC2", "ACC1", "ACC2"],
                "AccTo": ["ACC2", "ACC1", "ACC2", "ACC1"],
                "Quantity": [100.0, 50.0, 200.0, 150.0],
            }
        )
        result = _optimize_bank_transactions(df)

        prod1_result = result.filter(pl.col("ProductId") == "PROD1")
        prod2_result = result.filter(pl.col("ProductId") == "PROD2")

        assert len(prod1_result) == 1
        assert len(prod2_result) == 1

        assert prod1_result.row(0)[1:] == ("ACC1", "ACC2", 50.0)
        assert prod2_result.row(0)[1:] == ("ACC1", "ACC2", 50.0)

    def test_product_with_zero_net_flow(self):
        df = pl.DataFrame(
            {
                "ProductId": ["PROD1", "PROD1", "PROD2"],
                "AccFrom": ["ACC1", "ACC2", "ACC1"],
                "AccTo": ["ACC2", "ACC1", "ACC2"],
                "Quantity": [100.0, 100.0, 50.0],  # PROD1 cancels out
            }
        )
        result = _optimize_bank_transactions(df)

        assert len(result) == 1
        assert result.row(0) == ("PROD2", "ACC1", "ACC2", 50.0)


class TestValidateOptimization:
    def test_valid_optimization(self):
        original = pl.DataFrame(
            {"ProductId": ["PROD1", "PROD1"], "AccFrom": ["ACC1", "ACC2"], "AccTo": ["ACC2", "ACC1"], "Quantity": [150.0, 50.0]}
        )
        optimized = pl.DataFrame({"ProductId": ["PROD1"], "AccFrom": ["ACC1"], "AccTo": ["ACC2"], "Quantity": [100.0]})
        validate_optimization(original, optimized)

    def test_invalid_optimization_balance_mismatch(self):
        original = pl.DataFrame({"ProductId": ["PROD1"], "AccFrom": ["ACC1"], "AccTo": ["ACC2"], "Quantity": [100.0]})
        optimized = pl.DataFrame(
            {
                "ProductId": ["PROD1"],
                "AccFrom": ["ACC1"],
                "AccTo": ["ACC2"],
                "Quantity": [150.0],  # Wrong amount
            }
        )

        import pytest

        with pytest.raises(ValueError, match="Optimization validation failed"):
            validate_optimization(original, optimized)


class TestPropertyBasedTesting:
    def _generate_random_transactions(
        self, seed: int, num_products: int, num_accounts: int, num_transactions: int
    ) -> pl.DataFrame:
        np.random.seed(seed)

        products = [f"PROD{i}" for i in range(num_products)]
        accounts = [f"ACC{i}" for i in range(num_accounts)]

        data = []
        for _ in range(num_transactions):
            prod = np.random.choice(products)
            acc_from = np.random.choice(accounts)
            acc_to = np.random.choice(accounts)
            while acc_to == acc_from:  # Avoid self-transfers
                acc_to = np.random.choice(accounts)
            quantity = np.random.uniform(1, 1000)

            data.append({"ProductId": prod, "AccFrom": acc_from, "AccTo": acc_to, "Quantity": quantity})

        if not data:
            return pl.DataFrame(
                {"ProductId": [], "AccFrom": [], "AccTo": [], "Quantity": []},
                schema={"ProductId": pl.Utf8, "AccFrom": pl.Utf8, "AccTo": pl.Utf8, "Quantity": pl.Float64},
            )

        return pl.DataFrame(data)

    def test_optimization_preserves_balances(self):
        test_scenarios = [
            self._generate_random_transactions(seed=i, num_products=3, num_accounts=5, num_transactions=20) for i in range(10)
        ]

        for df in test_scenarios:
            if len(df) == 0:
                continue

            result = _optimize_bank_transactions(df)
            validate_optimization(df, result)

    def test_optimization_reduces_or_maintains_transaction_count(self):
        test_scenarios = [
            self._generate_random_transactions(seed=i, num_products=2, num_accounts=4, num_transactions=15) for i in range(10)
        ]

        for df in test_scenarios:
            result = _optimize_bank_transactions(df)
            assert len(result) <= len(df), "Optimization increased transaction count"


class TestEdgeCases:
    def test_very_large_quantities(self):
        df = pl.DataFrame(
            {"ProductId": ["PROD1", "PROD1"], "AccFrom": ["ACC1", "ACC2"], "AccTo": ["ACC2", "ACC1"], "Quantity": [1e15, 0.5e15]}
        )
        result = _optimize(df)
        assert len(result) == 1
        assert abs(result["Quantity"][0] - 0.5e15) < 1e10  # Within reasonable precision

    def test_very_small_quantities(self):
        df = pl.DataFrame(
            {"ProductId": ["PROD1", "PROD1"], "AccFrom": ["ACC1", "ACC2"], "AccTo": ["ACC2", "ACC1"], "Quantity": [1e-10, 0.9e-10]}
        )
        result = _optimize(df)
        # Should either cancel out or preserve small differences
        if len(result) > 0:
            assert result["Quantity"][0] > 0

    def test_many_accounts_star_pattern(self):
        # One central account connected to many others
        num_outer_accounts = 20
        df_data = []

        for i in range(num_outer_accounts):
            # Flow from center to outer
            df_data.append({"ProductId": "PROD1", "AccFrom": "ACC_CENTER", "AccTo": f"ACC_{i}", "Quantity": 100.0})
            # Smaller flow back to center
            df_data.append({"ProductId": "PROD1", "AccFrom": f"ACC_{i}", "AccTo": "ACC_CENTER", "Quantity": 30.0})

        df = pl.DataFrame(df_data)
        result = _optimize(df)

        # Should optimize to net flows only
        assert len(result) == num_outer_accounts  # One net flow per outer account
        for row in result.iter_rows(named=True):
            assert row["AccFrom"] == "ACC_CENTER"
            assert row["Quantity"] == 70.0  # Net flow


class TestEndToEndFlow:
    def test_large_dataset_performance(self):
        # Generate a large dataset
        large_dataset = self._generate_large_dataset(num_products=50, num_accounts=20, num_transactions=10000)

        import time

        start_time = time.time()
        result = _optimize_bank_transactions(large_dataset)
        end_time = time.time()

        processing_time = end_time - start_time
        assert processing_time < 30  # Should complete within 30 seconds

        # Should reduce transaction count significantly
        reduction_ratio = (len(large_dataset) - len(result)) / len(large_dataset)
        assert reduction_ratio > 0.1  # At least 10% reduction

        # Validate correctness
        validate_optimization(large_dataset, result)

        print(f"Processed {len(large_dataset)} transactions in {processing_time:.2f}s")
        print(f"Reduced to {len(result)} transactions ({reduction_ratio * 100:.1f}% reduction)")

    def test_memory_usage_large_dataset(self):
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate large dataset
        large_dataset = self._generate_large_dataset(num_products=100, num_accounts=50, num_transactions=50000)

        result = _optimize_bank_transactions(large_dataset)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 1GB for this dataset size)
        assert memory_increase < 1024, f"Memory usage increased by {memory_increase:.1f}MB"

        # Validate correctness
        validate_optimization(large_dataset, result)

        print(f"Memory increase: {memory_increase:.1f}MB for {len(large_dataset)} transactions")

    def test_parquet_file_integration(self):
        parquet_path = Path("tmp/transfers.parquet")

        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)

            # Ensure it has the expected columns
            expected_columns = {"ProductId", "AccFrom", "AccTo", "Quantity"}
            assert expected_columns.issubset(set(df.columns))

            original_count = len(df)
            result = _optimize_bank_transactions(df)
            optimized_count = len(result)

            assert optimized_count <= original_count
            validate_optimization(df, result)

            print(f"Parquet file: {original_count} -> {optimized_count} transactions")

    def _xml_to_bank_transfers(self, xml_content: str, accounts: dict) -> pl.DataFrame:
        root = ET.fromstring(xml_content)
        transfers = []

        # Group transfers by TransferId to create bank transfer pairs
        transfer_groups = {}
        for transfer in root.findall("Transfer"):
            transfer_id = transfer.find("TransferId").text
            if transfer_id not in transfer_groups:
                transfer_groups[transfer_id] = []

            transfer_groups[transfer_id].append(
                {
                    "ProductId": transfer.find("ProductId").text,
                    "PortfolioNumber": transfer.find("PortfolioNumber").text,
                    "Quantity": float(transfer.find("Quantity").text),
                    "Side": transfer.find("Side").text,
                }
            )

        # Convert to bank transfers
        for transfer_id, group in transfer_groups.items():
            if len(group) != 2:
                continue

            sell_transfer = next(t for t in group if t["Side"] == "SELL")
            buy_transfer = next(t for t in group if t["Side"] == "BUY")

            if sell_transfer["Quantity"] != buy_transfer["Quantity"]:
                continue

            acc_from = accounts[sell_transfer["PortfolioNumber"]]
            acc_to = accounts[buy_transfer["PortfolioNumber"]]

            transfers.append({"ProductId": sell_transfer["ProductId"], "AccFrom": acc_from, "AccTo": acc_to, "Quantity": sell_transfer["Quantity"]})

        return pl.DataFrame(transfers)

    def _generate_large_dataset(self, num_products: int, num_accounts: int, num_transactions: int) -> pl.DataFrame:
        import random

        products = [f"PROD_{i:04d}" for i in range(num_products)]
        accounts = [f"ACC_{i:04d}" for i in range(num_accounts)]

        data = []
        for _ in range(num_transactions):
            product = random.choice(products)
            acc_from = random.choice(accounts)
            acc_to = random.choice(accounts)
            while acc_to == acc_from:
                acc_to = random.choice(accounts)

            # Generate realistic quantities
            quantity = round(random.uniform(1, 10000), 2)

            data.append({"ProductId": product, "AccFrom": acc_from, "AccTo": acc_to, "Quantity": quantity})

        return pl.DataFrame(data)


class TestBenchmarking:
    def test_benchmark_current_algorithm(self):
        dataset_sizes = [100, 500, 1000, 5000]
        results = []

        for size in dataset_sizes:
            dataset = self._generate_benchmark_dataset(size)

            import time

            start_time = time.time()
            result = _optimize_bank_transactions(dataset)
            end_time = time.time()

            processing_time = end_time - start_time
            results.append((size, processing_time, len(result)))

            validate_optimization(dataset, result)

            print(f"Size {size}: {processing_time:.3f}s, {len(dataset)} -> {len(result)} transactions")

    def _generate_benchmark_dataset(self, size: int) -> pl.DataFrame:
        import random

        random.seed(42)  # Fixed seed for reproducible benchmarks

        num_products = max(5, size // 20)
        num_accounts = max(3, size // 50)

        products = [f"PROD_{i}" for i in range(num_products)]
        accounts = [f"ACC_{i}" for i in range(num_accounts)]

        data = []
        for i in range(size):
            product = random.choice(products)
            acc_from = random.choice(accounts)
            acc_to = random.choice(accounts)
            while acc_to == acc_from:
                acc_to = random.choice(accounts)

            quantity = round(random.uniform(10, 1000), 2)

            data.append({"ProductId": product, "AccFrom": acc_from, "AccTo": acc_to, "Quantity": quantity})

        return pl.DataFrame(data)


class TestRegressionSuite:
    def test_known_good_scenarios(self):
        scenarios = [
            {
                "name": "simple_bidirectional",
                "input": pl.DataFrame(
                    {"ProductId": ["PROD1", "PROD1"], "AccFrom": ["ACC1", "ACC2"], "AccTo": ["ACC2", "ACC1"], "Quantity": [100.0, 60.0]}
                ),
                "expected_output_count": 1,
                "expected_net_flow": 40.0,
            },
            {
                "name": "three_way_cycle",
                "input": pl.DataFrame(
                    {
                        "ProductId": ["PROD1", "PROD1", "PROD1"],
                        "AccFrom": ["ACC1", "ACC2", "ACC3"],
                        "AccTo": ["ACC2", "ACC3", "ACC1"],
                        "Quantity": [100.0, 100.0, 100.0],
                    }
                ),
                "expected_output_count": 0,  # Perfect cycle should cancel out
                "expected_net_flow": 0.0,
            },
            {
                "name": "multiple_products",
                "input": pl.DataFrame(
                    {
                        "ProductId": ["PROD1", "PROD1", "PROD2"],
                        "AccFrom": ["ACC1", "ACC2", "ACC1"],
                        "AccTo": ["ACC2", "ACC1", "ACC2"],
                        "Quantity": [150.0, 50.0, 200.0],
                    }
                ),
                "expected_output_count": 2,  # One per product
                "expected_net_flow": None,  # Don't check net flow for multi-product
            },
        ]

        for scenario in scenarios:
            result = _optimize_bank_transactions(scenario["input"])

            assert len(result) == scenario["expected_output_count"], (
                f"Scenario '{scenario['name']}': expected {scenario['expected_output_count']} outputs, got {len(result)}"
            )

            if scenario["expected_net_flow"] is not None and len(result) > 0:
                total_flow = result["Quantity"].sum()
                assert abs(total_flow - scenario["expected_net_flow"]) < 1e-10, (
                    f"Scenario '{scenario['name']}': expected net flow {scenario['expected_net_flow']}, got {total_flow}"
                )

            validate_optimization(scenario["input"], result)
