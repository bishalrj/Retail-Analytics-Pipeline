# Retail Intelligence & Market Basket Analysis Pipeline

## Project Title & Overview
This project, **Retail Intelligence & Market Basket Analysis Pipeline**, analyzes **1M+ transaction line items** from the Online Retail II dataset to drive revenue through two complementary analytics tracks:

1. **Customer segmentation** using RFM (Recency, Frequency, Monetary) to identify high-value customers and likely churn/hibernation cohorts.
2. **Product affinity discovery** using market basket association rules to enable targeted cross-selling and bundling.

## Business Problem
Retailers often struggle to identify:

- **High-value customers** among large customer bases.
- **Cross-selling opportunities** (items that tend to be purchased together) without manual curation.

This pipeline addresses that by converting raw transactional data into actionable customer segments and product recommendations derived directly from purchase behavior.

## Tech Stack
- **Python**
  - Pandas (ETL + transformations)
  - Scikit-learn (available for future modeling; current pipeline uses rule-based RFM)
  - MLxtend (Apriori + association rules)
  - Plotly (charts)
  - Streamlit (dashboard)
- **SQL**
  - SQLite (stores curated tables: `raw_data`, `clean_data`)
- **Excel IO**
  - OpenPyXL (reads `online_retail_II.xlsx`)

## The ETL Pipeline
1. **Ingestion (`scripts/ingestion.py`)**
   - Loads both Excel sheets: `Year 2009-2010` and `Year 2010-2011`
   - Concatenates them into one DataFrame
   - Cleans column names (lowercase, spaces -> underscores)
   - Writes the result into SQLite at `output/retail.db` in table `raw_data`

2. **Cleaning (`scripts/cleaning.py`)**
   - Removes rows where `customer_id` is null
   - Filters out **cancellations** where `invoice` starts with `C` (in this dataset: **19,494** cancellation rows filtered)
   - Keeps only rows where `quantity > 0` and `price > 0`
   - Creates `total_revenue = quantity * price`
   - Converts `invoice_date` to a datetime column
   - Saves the cleaned dataset back to SQLite as a new table: `clean_data`

## Advanced Analytics

### RFM Analysis
Customers are segmented using three behavioral measures computed from `clean_data`:

- **Recency**: days since last purchase (based on the maximum `invoice_date` in the dataset)
- **Frequency**: number of **unique invoices** per customer
- **Monetary**: total `total_revenue` per customer

Each metric is converted into **1–5 scores using quintiles** (`pd.qcut`). Scores are then mapped to customer segments:

- **Champions**
- **Loyal**
- **At Risk**
- **Hibernating**

The final segmentation output is saved to:
- `output/customer_segments.csv`

### Market Basket Analysis
Market basket analysis identifies products that customers frequently buy together using:

- **Apriori algorithm** (via **MLxtend**)
- **Minimum support**: `0.02`
- **Association rules metric**: `lift` (with `min_threshold=1`)
- **Filtering**: keep only rules with `confidence > 0.5`

Rules are generated from invoice-level baskets (one row per `invoice`, columns as products derived from `Description`). Lift and confidence support recommendations such as:

- “If product X is in the cart, product Y is likely to be purchased too.”

The resulting association rules are saved to:
- `output/market_basket_rules.csv`

## Actionable Insights (Business Recommendations)
Use the dashboard outputs to drive targeted revenue actions:

- **Target Hibernating customers** with win-back campaigns (discounts, personalized offers) and encourage their first re-purchase.
- **Prioritize Champions** with premium bundles and early access to new items to maximize retention and upsell.
- **Use Association Rules for bundling**:
  - When a customer is likely to buy product **X**, recommend frequently paired products **Y** using the highest-lift rules.
  - Create “complete the set” bundles where lift indicates meaningful affinity.

## How to Run

### 1) Install dependencies
From the project root (`churn/`):
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) Run the pipeline scripts (ETL + analytics)
```powershell
.\venv\Scripts\python.exe scripts\ingestion.py
.\venv\Scripts\python.exe scripts\cleaning.py
.\venv\Scripts\python.exe scripts\rfm.py
.\venv\Scripts\python.exe scripts\market_basket.py
```

### 3) Start the dashboard
```powershell
.\venv\Scripts\python.exe -m streamlit run app\dashboard.py
```

## Outputs
- `output/retail.db`
  - `raw_data`
  - `clean_data`
- `output/customer_segments.csv`
- `output/market_basket_rules.csv`

## Hardware Context
The pipeline is designed to run on consumer hardware. With the provided dataset and settings, it is expected to work comfortably on:

- CPU: i5
- RAM: 16GB
- GPU: RTX 3050 (not required for the core analytics; primarily for general dev convenience)

