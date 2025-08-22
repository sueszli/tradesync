import json
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl
from loguru import logger


def validate_xml_structure(transfers_path: Path):
    """Validate XML transfer file structure and required fields.

    Args:
        transfers_path: XML file with transfer records

    Raises:
        ValueError: Invalid XML structure or missing required fields
    """
    try:
        tree = ET.parse(transfers_path)
        root = tree.getroot()

        transfers = root.findall("Transfer")
        if not transfers:
            raise ValueError("No 'Transfer' elements found in XML")

        required_fields = ["ProductId", "PortfolioNumber", "Quantity", "Side", "TransferId"]
        for i, t in enumerate(transfers):
            for field in required_fields:
                element = t.find(field)
                if element is None:
                    raise ValueError(f"Transfer {i}: Missing required field '{field}'")
                if not element.text:
                    raise ValueError(f"Transfer {i}: Field '{field}' is empty")

            try:
                float(t.find("Quantity").text)
            except ValueError:
                raise ValueError(f"Transfer {i}: 'Quantity' must be a number, got: {t.find('Quantity').text}")

            side = t.find("Side").text
            if side not in ["BUY", "SELL"]:
                raise ValueError(f"Transfer {i}: 'Side' must be 'BUY' or 'SELL', got: {side}")

    except ET.ParseError as e:
        raise ValueError(f"Invalid XML format: {e}")
    logger.info(f"Validated XML structure for: {transfers_path}")


def validate_json_structure(accounts_path: Path):
    """Validate JSON accounts mapping structure.

    Args:
        accounts_path: JSON file with {portfolio_name: account_id} mapping

    Raises:
        ValueError: Invalid JSON structure or empty/invalid mappings
    """
    try:
        data = json.loads(accounts_path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Accounts JSON must be an object/dictionary")

        if not data:
            raise ValueError("Accounts JSON cannot be empty")

        for portfolio, account in data.items():
            if not isinstance(portfolio, str) or not portfolio:
                raise ValueError(f"Portfolio name must be non-empty string, got: {portfolio}")
            if not isinstance(account, str) or not account:
                raise ValueError(f"Account for portfolio '{portfolio}' must be non-empty string, got: {account}")

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    logger.info(f"Validated JSON structure for: {accounts_path}")


def validate_optimization(original_df: pl.DataFrame, optimized_df: pl.DataFrame):
    """Validate optimization preserves net account balances and total magnitude.

    Args:
        original_df: Original transfer flows [ProductId, AccFrom, AccTo, Quantity]
        optimized_df: Optimized transfer flows with same schema

    Raises:
        ValueError: Net flows don't match or total magnitude changed
    """
    # validate that net flows (net balance of any account for any product) are preserved after optimization
    original_balances = _calc_net_balance_per_account(original_df)
    optimized_balances = _calc_net_balance_per_account(optimized_df)
    comparison = (
        original_balances.join(optimized_balances, on=["ProductId", "AccFrom"], how="full", suffix="_opt")
        .with_columns(
            [
                pl.col("NetBalance").fill_null(0),
                pl.col("NetBalance_opt").fill_null(0),
            ]
        )
        .with_columns([(pl.col("NetBalance") - pl.col("NetBalance_opt")).abs().alias("Difference")])
        .filter(pl.col("Difference") > 1e-8)
        .sort(["ProductId", "AccFrom"])
    )
    if len(comparison):
        raise ValueError(f"Optimization validation failed: net flows do not match for {len(comparison)} accounts.")

    # validate magnitude preservation
    orig_total = original_balances["NetBalance"].abs().sum()
    opt_total = optimized_balances["NetBalance"].abs().sum()
    if abs(orig_total - opt_total) > 1e-6:
        raise ValueError(f"Total balance magnitude changed from {orig_total:.6f} to {opt_total:.6f}")

    logger.info("Net account balances are preserved correctly!")


def _calc_net_balance_per_account(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate net balance per (product, account) pair from transfer flows.

    Args:
        df: Transfer flows [ProductId, AccFrom, AccTo, Quantity]

    Returns:
        Net balances [ProductId, AccFrom, NetBalance] for non-zero accounts
    """
    from_accounts = df.select(["ProductId", pl.col("AccFrom").alias("Account")]).unique()
    to_accounts = df.select(["ProductId", pl.col("AccTo").alias("Account")]).unique()
    all_accounts = pl.concat([from_accounts, to_accounts]).unique()

    out_flows = df.group_by(["ProductId", "AccFrom"]).agg(pl.col("Quantity").sum().alias("OutFlow")).rename({"AccFrom": "Account"})
    in_flows = df.group_by(["ProductId", "AccTo"]).agg(pl.col("Quantity").sum().alias("InFlow")).rename({"AccTo": "Account"})

    return (
        all_accounts.join(out_flows, on=["ProductId", "Account"], how="left")
        .join(in_flows, on=["ProductId", "Account"], how="left")
        .with_columns(
            [
                pl.col("OutFlow").fill_null(0),
                pl.col("InFlow").fill_null(0),
            ]
        )
        .with_columns([(pl.col("InFlow") - pl.col("OutFlow")).alias("NetBalance")])
        .filter(pl.col("NetBalance").abs() > 1e-10)
        .select(["ProductId", pl.col("Account").alias("AccFrom"), "NetBalance"])
        .sort(["ProductId", "AccFrom"])
    )
