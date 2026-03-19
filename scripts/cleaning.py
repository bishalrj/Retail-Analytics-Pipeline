from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


RAW_TABLE = "raw_data"
CLEAN_TABLE = "clean_data"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / "output" / "retail.db"

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {RAW_TABLE}", engine)

        # Align with your requested column name.
        if "invoice_date" not in df.columns and "invoicedate" in df.columns:
            df = df.rename(columns={"invoicedate": "invoice_date"})

        # 1) Remove rows where customer_id is null.
        if "customer_id" not in df.columns:
            raise KeyError("Expected column 'customer_id' in raw_data")
        df = df[df["customer_id"].notna()].copy()

        # 2) Remove cancellations: invoices starting with 'C'.
        if "invoice" not in df.columns:
            raise KeyError("Expected column 'invoice' in raw_data")
        invoice_str = df["invoice"].astype(str)
        df = df[~invoice_str.str.startswith("C")].copy()

        # 3) Ensure quantity and price are both > 0.
        for col in ("quantity", "price"):
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' in raw_data")
        df = df[(df["quantity"] > 0) & (df["price"] > 0)].copy()

        # 4) Add total_revenue = quantity * price.
        df["total_revenue"] = df["quantity"] * df["price"]

        # 5) Convert invoice_date to datetime.
        if "invoice_date" not in df.columns:
            raise KeyError("Expected column 'invoice_date' in raw_data (or 'invoicedate')")
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

        df.to_sql(
            CLEAN_TABLE,
            engine,
            if_exists="replace",
            index=False,
            chunksize=10_000,
        )

        print(f"Cleaned row count: {len(df)}")
        print(df.head(5))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

