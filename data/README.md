# Data

This project uses a deterministic synthetic single-symbol limit-order-book dataset for educational and research purposes.

## Required files

Place the following files in `data/raw/` before running the analysis:

- `AI_Order_Execution.xlsx`
- `Single_Symbol_LOB_History_50_10000.json.gz`

The Excel workbook contains order-level observations, prediction targets, chronological data splits, a data dictionary, and a description of the historical JSON structure.

The compressed JSON file contains 50 pre-submission limit-order-book snapshots for each order.

## Data availability

The raw data are not currently included in this repository. This avoids redistributing course materials before their sharing permissions have been confirmed.

## Important notice

The dataset is synthetic teaching data. It is not observed Nasdaq or LOBSTER data and should not be interpreted as evidence about real AAPL trading or execution.
