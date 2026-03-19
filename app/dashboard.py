from __future__ import annotations

from pathlib import Path

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "output" / "retail.db"
SEGMENTS_CSV_PATH = REPO_ROOT / "output" / "customer_segments.csv"
MARKET_BASKET_RULES_CSV_PATH = REPO_ROOT / "output" / "market_basket_rules.csv"

INPUT_TABLE = "clean_data"


def _format_currency(x: float | int) -> str:
    return f"£{float(x):,.2f}"


@st.cache_data(show_spinner=False)
def load_countries() -> list[str]:
    if not DB_PATH.exists():
        return []
    engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")
    try:
        countries = pd.read_sql_query(
            f"SELECT DISTINCT country FROM {INPUT_TABLE} ORDER BY country",
            engine,
        )["country"].dropna().astype(str).tolist()
        return countries
    finally:
        engine.dispose()


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame:
    if not SEGMENTS_CSV_PATH.exists():
        return pd.DataFrame(
            columns=["customer_id", "segment", "r_score", "f_score", "m_score", "recency_days", "frequency", "monetary"]
        )

    seg = pd.read_csv(SEGMENTS_CSV_PATH)
    # Align with how `customer_segments.csv` represents ids.
    seg["customer_id"] = seg["customer_id"].astype(float).astype(int).astype(str)
    return seg


@st.cache_data(show_spinner=False)
def load_metrics_by_country(country: str) -> dict[str, float]:
    if not DB_PATH.exists():
        return {"total_revenue": 0.0, "total_customers": 0, "avg_order_value": 0.0}

    engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")
    try:
        where = "" if country == "All" else "WHERE country = :country"
        params = {} if country == "All" else {"country": country}

        q = f"""
            SELECT
                SUM(total_revenue) AS total_revenue,
                COUNT(DISTINCT customer_id) AS total_customers,
                CAST(SUM(total_revenue) AS FLOAT) / NULLIF(COUNT(DISTINCT invoice), 0) AS avg_order_value
            FROM {INPUT_TABLE}
            {where}
        """
        out = pd.read_sql_query(q, engine, params=params).iloc[0].to_dict()
        out["total_revenue"] = float(out["total_revenue"] or 0.0)
        out["total_customers"] = int(out["total_customers"] or 0)
        out["avg_order_value"] = float(out["avg_order_value"] or 0.0)
        return out
    finally:
        engine.dispose()


@st.cache_data(show_spinner=False)
def load_top_products_by_country(country: str, limit: int = 10) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["stockcode", "description", "revenue"])

    engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")
    try:
        where = "" if country == "All" else "WHERE country = :country"
        params = {} if country == "All" else {"country": country}

        q = f"""
            SELECT
                stockcode,
                MAX(description) AS description,
                SUM(total_revenue) AS revenue
            FROM {INPUT_TABLE}
            {where}
            GROUP BY stockcode
            ORDER BY revenue DESC
            LIMIT :limit
        """
        out = pd.read_sql_query(q, engine, params={**params, "limit": limit})
        out["description"] = out["description"].astype(str)
        out["revenue"] = out["revenue"].astype(float)
        return out
    finally:
        engine.dispose()


@st.cache_data(show_spinner=False)
def load_customer_ids_in_country(country: str) -> list[str]:
    if country == "All":
        # For "All" we will not need a filter list.
        return []

    engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")
    try:
        q = f"SELECT DISTINCT customer_id FROM {INPUT_TABLE} WHERE country = :country"
        ids = pd.read_sql_query(q, engine, params={"country": country})["customer_id"].dropna()
        # SQLite stored this as FLOAT. Normalize to integer-like strings.
        ids = ids.astype(float).astype(int).astype(str)
        return ids.tolist()
    finally:
        engine.dispose()


def _parse_itemset_str(x: object) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    # Itemsets were created by joining with `", "`. We split on commas defensively.
    return [part.strip() for part in re.split(r"\s*,\s*", s) if part.strip()]


@st.cache_data(show_spinner=False)
def load_market_basket_rules() -> pd.DataFrame:
    if not MARKET_BASKET_RULES_CSV_PATH.exists():
        return pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift", "antecedent_items", "consequent_items"]
        )

    rules = pd.read_csv(MARKET_BASKET_RULES_CSV_PATH)
    required = {"antecedents", "consequents", "support", "confidence", "lift"}
    missing = required - set(rules.columns)
    if missing:
        raise KeyError(f"market_basket_rules.csv missing columns: {sorted(missing)}")

    for col in ("support", "confidence", "lift"):
        rules[col] = pd.to_numeric(rules[col], errors="coerce")

    rules["antecedent_items"] = rules["antecedents"].apply(_parse_itemset_str)
    rules["consequent_items"] = rules["consequents"].apply(_parse_itemset_str)
    return rules


def set_dark_theme() -> None:
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: #e6edf3; }
        .stApp { background-color: #0e1117; }
        .stMetric { background-color: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Retail Churn - RFM Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    set_dark_theme()

    st.title("Retail Customer Segments (RFM)")
    st.caption("KPIs, segments, and top products from `output/retail.db`.")

    countries = load_countries()
    selected_country = st.sidebar.selectbox("Country", ["All"] + countries)

    metrics = load_metrics_by_country(selected_country)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", _format_currency(metrics["total_revenue"]))
    col2.metric("Total Customers", f"{metrics['total_customers']:,d}")
    col3.metric("Average Order Value", _format_currency(metrics["avg_order_value"]))

    segments = load_segments()
    if segments.empty:
        st.warning("`output/customer_segments.csv` not found or empty. Run `python scripts/rfm.py`.")
        return

    if selected_country == "All":
        seg_filtered = segments
    else:
        ids = load_customer_ids_in_country(selected_country)
        seg_filtered = segments[segments["customer_id"].isin(ids)].copy()

    seg_counts = (
        seg_filtered.groupby("segment", as_index=False)
        .size()
        .rename(columns={"size": "customers"})
        .sort_values("customers", ascending=False)
    )

    # Pie chart for segments.
    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=seg_counts["segment"],
                values=seg_counts["customers"],
                hole=0.45,
                textinfo="label+percent",
            )
        ]
    )
    pie_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Segment",
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # Top 10 products bar chart by revenue.
    top_products = load_top_products_by_country(selected_country, limit=10)
    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=top_products["stockcode"],
                y=top_products["revenue"],
                hovertext=top_products["description"],
                marker_color="#5dade2",
            )
        ]
    )
    bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Stockcode",
        yaxis_title="Revenue",
    )

    tab_overview, tab_product_strategy = st.tabs(["Customer Analytics", "Product Strategy"])

    with tab_overview:
        left, right = st.columns([1, 2])
        with left:
            st.subheader("Segment Distribution")
            st.plotly_chart(pie_fig, use_container_width=True)
        with right:
            st.subheader("Top 10 Products by Revenue")
            st.plotly_chart(bar_fig, use_container_width=True)

    with tab_product_strategy:
        st.subheader("Recommendation Engine")

        rules = load_market_basket_rules()
        if rules.empty:
            st.warning("`output/market_basket_rules.csv` not found or empty. Run `python scripts/market_basket.py` first.")
        else:
            # Collect all products appearing in antecedents/consequents.
            all_products = sorted(
                set(sum(rules["antecedent_items"].tolist(), []) + sum(rules["consequent_items"].tolist(), []))
            )
            if not all_products:
                st.warning("No product names were parsed from `market_basket_rules.csv`.")
            else:
                selected_product = st.selectbox("Select a product", all_products)

                mask = rules["antecedent_items"].apply(lambda items: selected_product in items)
                matched = rules[mask].copy()

                if matched.empty:
                    st.info("No association rules found for this product as an antecedent (left-hand side).")
                else:
                    # Build recommendations from consequents of all matching rules.
                    exploded = matched[["consequent_items", "support", "confidence", "lift"]].explode("consequent_items")
                    exploded = exploded.rename(columns={"consequent_items": "product"})

                    recs = (
                        exploded.groupby("product", as_index=False)
                        .agg(
                            best_support=("support", "max"),
                            best_confidence=("confidence", "max"),
                            best_lift=("lift", "max"),
                        )
                        .sort_values(["best_lift", "best_confidence"], ascending=[False, False])
                    )

                    st.caption(f"Matched {len(matched)} rule(s). Recommendations from their consequents.")
                    st.dataframe(
                        recs.rename(columns={"product": "Frequently Bought Together Product"})
                        .head(10),
                        use_container_width=True,
                    )

        st.divider()
        st.subheader("Rules: Support vs Confidence (Lift = marker size)")

        if not rules.empty:
            # Scale marker size for readability.
            lift_values = rules["lift"].fillna(0).clip(lower=0)
            marker_sizes = (lift_values * 3).astype(float)

            scatter = go.Figure(
                data=[
                    go.Scatter(
                        x=rules["support"],
                        y=rules["confidence"],
                        mode="markers",
                        marker=dict(size=marker_sizes, color=rules["lift"], colorscale="Blues", showscale=True, opacity=0.85),
                        text=[f"{a} -> {c}" for a, c in zip(rules["antecedents"], rules["consequents"])],
                        hovertemplate=(
                            "Antecedents=%{text}<br>"
                            "Support=%{x:.4f}<br>"
                            "Confidence=%{y:.4f}<br>"
                            "Lift=%{marker.color:.2f}<extra></extra>"
                        ),
                    )
                ]
            )
            scatter.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(scatter, use_container_width=True)

        st.markdown(
            "Lift measures how much more likely the consequent is to occur with the antecedent than it would be by random chance. "
            "For example, a lift of 5 means these items are **5x more likely** to be bought together than by random chance."
        )


if __name__ == "__main__":
    main()

