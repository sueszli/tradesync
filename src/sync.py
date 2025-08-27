import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import click
import networkx as nx
import polars as pl
from loguru import logger
from tqdm import tqdm

from validate import validate_json_structure, validate_optimization, validate_xml_structure

ZERO_THRESHOLD = 1e-10

EMPTY_RESULT = pl.DataFrame(
    {"ProductId": [], "AccFrom": [], "AccTo": [], "Quantity": []},
    schema={"ProductId": pl.Utf8, "AccFrom": pl.Utf8, "AccTo": pl.Utf8, "Quantity": pl.Float64},
)


def _dump_to_xml(df: pl.DataFrame, output: Path) -> None:
    df = df.with_columns(pl.lit(str(uuid.uuid4())).alias("TransferId"))
    output.write_text(df.to_pandas().to_xml(root_name="Transfers", row_name="Transfer", index=False))


def _decycle(df: pl.DataFrame, search_depth=3) -> pl.DataFrame:
    if len(df) <= 2:
        return df

    flows = {(row["AccFrom"], row["AccTo"]): row["Quantity"] for row in df.iter_rows(named=True)}
    G = nx.DiGraph()
    G.add_weighted_edges_from([(acc_from, acc_to, quantity) for (acc_from, acc_to), quantity in flows.items()])

    max_cycle_length = min(search_depth, len(G.nodes))
    cycles_to_process = {tuple(sorted(cycle)): cycle for cycle in nx.simple_cycles(G, length_bound=max_cycle_length)}

    for cycle in sorted(cycles_to_process.values(), key=len):
        cycle_flows = [flows.get((cycle[i], cycle[(i + 1) % len(cycle)]), 0) for i in range(len(cycle))]
        min_flow = min(cycle_flows)
        if min_flow <= ZERO_THRESHOLD:
            continue
        for i in range(len(cycle)):
            edge_key = (cycle[i], cycle[(i + 1) % len(cycle)])
            if not edge_key in flows:
                continue
            flows[edge_key] -= min_flow

    product_id = df["ProductId"][0]
    result_data = [
        {"ProductId": product_id, "AccFrom": acc_from, "AccTo": acc_to, "Quantity": quantity}
        for (acc_from, acc_to), quantity in flows.items()
        if quantity > ZERO_THRESHOLD
    ]
    return pl.DataFrame(result_data) if result_data else EMPTY_RESULT


def _optimize(product_df: pl.DataFrame) -> pl.DataFrame:
    if len(product_df) == 0:
        return product_df

    # aggregate parallel (A -> B) flows
    aggregated = (
        product_df.filter(pl.col("AccFrom") != pl.col("AccTo"))  # drop cyclic A -> A flows
        .group_by(["AccFrom", "AccTo"])
        .agg(pl.col("Quantity").sum())
        .filter(pl.col("Quantity") > ZERO_THRESHOLD)
    )
    if len(aggregated) == 0:
        return EMPTY_RESULT

    # aggregate bidirectional (A <-> B) flows
    product_id = product_df["ProductId"][0]
    one_directional = (
        aggregated.with_columns(
            [
                pl.min_horizontal([pl.col("AccFrom"), pl.col("AccTo")]).alias("Account1"),
                pl.max_horizontal([pl.col("AccFrom"), pl.col("AccTo")]).alias("Account2"),
                # 1 if forward (from < to), -1 if reverse - used as factor for signed quantity
                pl.when(pl.col("AccFrom") < pl.col("AccTo")).then(1).otherwise(-1).alias("Direction"),
            ]
        )
        .with_columns([(pl.col("Quantity") * pl.col("Direction")).alias("SignedQuantity")])
        .group_by(["Account1", "Account2"])
        .agg(pl.col("SignedQuantity").sum().alias("NetQuantity"))
        .filter(pl.col("NetQuantity").abs() > ZERO_THRESHOLD)
        .with_columns(
            [
                pl.when(pl.col("NetQuantity") > 0).then(pl.col("Account1")).otherwise(pl.col("Account2")).alias("AccFrom"),
                pl.when(pl.col("NetQuantity") > 0).then(pl.col("Account2")).otherwise(pl.col("Account1")).alias("AccTo"),
                pl.col("NetQuantity").abs().alias("Quantity"),
                pl.lit(product_id).alias("ProductId"),
            ]
        )
        .select(["ProductId", "AccFrom", "AccTo", "Quantity"])
    )
    if len(one_directional) == 0:
        return EMPTY_RESULT

    return _decycle(one_directional)


def _optimize_bank_transactions(df: pl.DataFrame) -> pl.DataFrame:
    if len(df) == 0:
        return df
    results = [
        optimized
        for product_data in tqdm(df.group_by("ProductId"), total=len(df["ProductId"].unique()))
        if len(optimized := _optimize(product_data[1])) > 0
    ]
    return pl.concat(results) if results else EMPTY_RESULT


def _get_bank_transfers(transfers: Path, accounts: Path) -> pl.DataFrame:
    # join tables on portfolio name as key
    accounts = pl.LazyFrame(
        [{"PortfolioNumber": portfolio, "Account": account} for portfolio, account in json.loads(accounts.read_text()).items()]
    )
    cols = ["ProductId", "PortfolioNumber", "Side", "TransferId", "Quantity"]
    transfers_parsed = [{col: t.find(col).text for col in cols} for t in ET.parse(transfers).getroot().findall("Transfer")]
    transfers = pl.LazyFrame(transfers_parsed).with_columns(pl.col("Quantity").cast(pl.Float64))
    joined = transfers.join(accounts, on="PortfolioNumber", how="left")

    # reconstruct (SELL, BUY) bank transfer pairs
    return (
        joined.group_by(["TransferId", "ProductId"]).agg(
            [
                pl.col("Side").n_unique().alias("PairCount"),
                pl.col("Account").filter(pl.col("Side") == "SELL").first().alias("AccFrom"),
                pl.col("Account").filter(pl.col("Side") == "BUY").first().alias("AccTo"),
                pl.col("Quantity").first().alias("Quantity"),
            ]
        )
    ).collect()


@click.command()
@click.option("--transfers", type=click.Path(exists=True, path_type=Path), help="Path to the transfers XML file.")
@click.option("--accounts", type=click.Path(exists=True, path_type=Path), help="Path to the accounts JSON file.")
@click.option("--output", type=click.Path(path_type=Path), help="Path to the output XML file.")
@click.option("--pedantic", is_flag=True, help="Enable expensive sanity checks.")
def run(transfers: Path, accounts: Path, output: Path, pedantic: bool):
    if not transfers.suffix.lower() == ".xml":
        raise ValueError(f"Transfers file must be XML, got: {transfers.suffix}")
    if not accounts.suffix.lower() == ".json":
        raise ValueError(f"Accounts file must be JSON, got: {accounts.suffix}")
    if not output.suffix.lower() == ".xml":
        raise ValueError(f"Output file must be XML, got: {output.suffix}")
    if pedantic:
        validate_xml_structure(transfers)
        validate_json_structure(accounts)
    logger.info(f"Files loaded")

    original = _get_bank_transfers(transfers, accounts)
    logger.info(f"Found {original.height} initial bank transfers")

    optimized = _optimize_bank_transactions(original)
    logger.info(
        f"Optimized to {optimized.height} bank transfers (Removed {(original.height - optimized.height) / original.height * 100:.2f}%)"
    )

    if pedantic:
        validate_optimization(original, optimized)

    _dump_to_xml(optimized, output)


if __name__ == "__main__":
    run()
