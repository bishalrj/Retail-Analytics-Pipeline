from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


DB_PATH = "output/retail.db"
INPUT_TABLE = "clean_data"
OUTPUT_CSV = "output/customer_segments.csv"


def _score_quintiles(series: pd.Series) -> pd.Series:
    """
    Create 1..5 quintile scores using pd.qcut.
    Higher values -> higher scores (1=lowest, 5=highest).
    """
    # Use rank to avoid qcut collapsing bins due to repeated values.
    # This keeps us aligned with the requested 1..5 scoring.
    ranked = series.rank(method="first")
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / DB_PATH
    out_csv_path = repo_root / OUTPUT_CSV

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")

    try:
        df = pd.read_sql_query(f"SELECT * FROM {INPUT_TABLE}", engine)

        if "customer_id" not in df.columns:
            raise KeyError("Expected 'customer_id' column in clean_data")
        if "invoice" not in df.columns:
            raise KeyError("Expected 'invoice' column in clean_data")
        if "total_revenue" not in df.columns:
            raise KeyError("Expected 'total_revenue' column in clean_data")
        if "invoice_date" not in df.columns:
            raise KeyError("Expected 'invoice_date' column in clean_data")

        df = df[df["customer_id"].notna()].copy()
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df = df[df["invoice_date"].notna()].copy()

        # Normalize customer_id to a stable string (SQLite stored it as FLOAT).
        # This avoids "13085.0" vs "13085" issues when saving CSV.
        df["customer_id"] = df["customer_id"].astype(float).astype(int).astype(str)

        max_date = df["invoice_date"].max()
        df["recency_days"] = (max_date - df["invoice_date"]).dt.days

        rfm = (
            df.groupby("customer_id", as_index=False)
            .agg(
                recency_days=("recency_days", "min"),
                frequency=("invoice", pd.Series.nunique),
                monetary=("total_revenue", "sum"),
            )
        )

        # R (recency): smaller days -> more recent -> higher score.
        # Quintile labels come from smallest -> largest, so invert with [5..1].
        rfm["r_score"] = pd.qcut(
            rfm["recency_days"].rank(method="first"),
            5,
            labels=[5, 4, 3, 2, 1],
        ).astype(int)
        rfm["f_score"] = _score_quintiles(rfm["frequency"])
        rfm["m_score"] = _score_quintiles(rfm["monetary"])

        # Segment mapping:
        # - Champions: high recency + high frequency (using R & F)
        # - Loyal: high recency + low/moderate frequency
        # - At Risk: low/moderate recency + high frequency
        # - Hibernating: low recency + low/moderate frequency
        rfm["segment"] = "Hibernating"
        rfm.loc[(rfm["r_score"] >= 4) & (rfm["f_score"] >= 4), "segment"] = "Champions"
        rfm.loc[(rfm["r_score"] >= 4) & (rfm["f_score"] < 4), "segment"] = "Loyal"
        rfm.loc[(rfm["r_score"] < 4) & (rfm["f_score"] >= 4), "segment"] = "At Risk"

        # Save to CSV for downstream analysis/visualization.
        rfm = rfm.sort_values(["segment", "r_score", "f_score", "m_score"], ascending=[True, False, False, False])
        rfm.to_csv(out_csv_path, index=False)

        print(f"Wrote: {out_csv_path}")
        print(f"Customer rows: {len(rfm)}")
        print(rfm.head(5))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

