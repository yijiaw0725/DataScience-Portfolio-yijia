# Data Engineering ETL Pipeline with DuckDB (Construction Data)

## Project Overview
This project demonstrates a **modern data engineering pipeline** using the **Bronze–Silver–Gold** architecture with **DuckDB** and Python.  
The goal is to simulate a realistic ETL process, transforming raw operational data into clean, analytics-ready datasets.

---

## Architecture: Bronze–Silver–Gold
- **Bronze (Raw Layer)**  
  Store raw ingested CSV files exactly as received.  
  Benefits: full reproducibility, traceability.

- **Silver (Clean Layer)**  
  Standardize column names, data types, and join keys.  
  Create stable `dim_person` and clean `work_orders` views.  
  Benefits: consistent formats, improved join quality.

- **Gold (Business Layer)**  
  Aggregated, business-ready metrics such as valid move-in issue counts, completion KPIs, etc.  
  Benefits: directly usable for BI dashboards and reporting.

---
## Tools & Skills
- **DuckDB**: In-process analytical database for SQL transformations.
- **Python**: Data wrangling, pipeline orchestration.
- **Pandas**: Initial CSV inspection and sampling.
- **Colab**: Cloud environment for development.
- **GitHub**: Version control and portfolio hosting.

---

## Sample Data
- `people_sample.csv`: Simulated bonus master employee list.
- `work_orders_sample.csv`: Simulated move-in issues dataset.

---
