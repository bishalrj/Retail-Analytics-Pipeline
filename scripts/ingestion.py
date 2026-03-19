from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


SHEETS = ["Year 2009-2010", "Year 2010-2011"]
DB_TABLE = "raw_data"


def clean_column_name(name: object) -> str:
    s = str(name).strip().lower()
    # Replace any whitespace run with a single underscore.
    s = re.sub(r"\s+", "_", s)
    return s


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    excel_path = repo_root / "data" / "online_retail_II.xlsx"
    db_path = repo_root / "output" / "retail.db"

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df_parts: list[pd.DataFrame] = []
    for sheet in SHEETS:
        df_parts.append(pd.read_excel(excel_path, sheet_name=sheet))

    df = pd.concat(df_parts, ignore_index=True)
    df.columns = [clean_column_name(c) for c in df.columns]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    df.to_sql(DB_TABLE, engine, if_exists="replace", index=False)
    engine.dispose()

    print(f"Final row count: {len(df)}")
    print(df.head(5))

    # Optional sanity-check that the data is actually in SQLite.
    # (Keeps the verification logic close to the ingestion step.)
    with sqlite3.connect(db_path.as_posix()) as con:
        cnt = con.execute(f"SELECT COUNT(*) FROM {DB_TABLE}").fetchone()[0]
        sample = pd.read_sql_query(f"SELECT * FROM {DB_TABLE} LIMIT 5", con)
    assert int(cnt) == len(df), "Row count mismatch between DataFrame and SQLite"
    assert len(sample) == 5, "Expected 5-row sample from SQLite"


if __name__ == "__main__":
    main()

