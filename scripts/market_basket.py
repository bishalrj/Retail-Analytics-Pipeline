from __future__ import annotations

from pathlib import Path

import warnings

import pandas as pd
from sqlalchemy import create_engine

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


DB_PATH = "output/retail.db"
INPUT_TABLE = "clean_data"
OUTPUT_CSV = "output/market_basket_rules.csv"


def _itemset_to_str(x: object) -> str:
    # mlxtend returns frozenset(...) for antecedents/consequents
    if x is None:
        return ""
    if isinstance(x, (set, frozenset, list, tuple)):
        return ", ".join(map(str, list(x)))
    return str(x)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / DB_PATH
    out_csv_path = repo_root / OUTPUT_CSV

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    try:
        df = pd.read_sql_query(
            f"""
            SELECT invoice, description
            FROM {INPUT_TABLE}
            WHERE description IS NOT NULL
            """,
            engine,
        )

        if df.empty:
            raise ValueError("No rows found in clean_data (after filtering description IS NOT NULL).")

        # Build transactions: one "basket" per invoice containing unique descriptions.
        grouped = df.groupby("invoice")["description"].apply(lambda s: sorted(set(s.dropna().astype(str))))
        invoice_ids = grouped.index.astype(str).tolist()
        transactions = grouped.tolist()

        # pandas emits a FutureWarning when constructing SparseDtype with arbitrary fill values.
        # It does not affect results; we suppress it to keep CLI output clean.
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"Allowing arbitrary scalar fill_value in SparseDtype is deprecated.*",
        )

        # One-hot / basket encoding (kept sparse to reduce memory).
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions, sparse=True)

        # Convert to boolean so the SparseDtype fill_value is valid (avoids pandas FutureWarning).
        te_ary_bool = te_ary.astype(bool)
        basket = pd.DataFrame.sparse.from_spmatrix(
            te_ary_bool,
            index=invoice_ids,
            columns=[str(c) for c in te.columns_],
        )

        # Apriori on item presence.
        frequent_itemsets = apriori(
            basket,
            min_support=0.02,
            use_colnames=True,
            low_memory=True,
        )

        # Association rules using lift.
        rules = association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=1,
        )

        rules = rules[rules["confidence"] > 0.5].copy()

        # Save only requested fields.
        out = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
        out["antecedents"] = out["antecedents"].map(_itemset_to_str)
        out["consequents"] = out["consequents"].map(_itemset_to_str)

        out.to_csv(out_csv_path, index=False)

        print(f"Frequent itemsets: {len(frequent_itemsets)}")
        print(f"Rules after filters: {len(out)}")
        print(f"Wrote: {out_csv_path}")
        print(out.head(5))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

