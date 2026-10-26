# Data setup and provenance

This project uses deterministic synthetic teaching data for one symbol (AAPL). Raw files are intentionally excluded from the repository until their redistribution permissions are confirmed.

> [!WARNING]
> The data are not observed Nasdaq or LOBSTER records and must not be used to make claims about real AAPL execution quality.

## Required local files

Place exactly these two files in `data/raw/`:

```text
data/raw/
├── AI_Order_Execution.xlsx
└── Single_Symbol_LOB_History_50_10000.json.gz
```

If the supplied workbook is named `Single_Symbol_LOB_Order_Dataset_10000.xlsx`, make a local copy named `AI_Order_Execution.xlsx` before running the notebooks. Do not keep multiple `.xlsx` or history `.json`/`.json.gz` files in `data/raw/` because the current notebooks discover inputs by file type.

`data/raw/` is ignored by Git and should remain local.

## Expected contents

| File | Expected contents |
|---|---|
| `AI_Order_Execution.xlsx` | 10,000 order-level rows, 72 analysis columns, target eligibility flags, chronological split labels, data dictionary, JSON schema, summary, and quality checks |
| `Single_Symbol_LOB_History_50_10000.json.gz` | 50 pre-submission order-book snapshots per order (500,000 snapshots in total) |

The workbook contains six sheets: `README`, `Orders`, `Data_Dictionary`, `JSON_Schema`, `Summary`, and `Quality_Checks`.

## Chronological split

| Split | Trading days | Orders |
|---|---:|---:|
| Train | first 14 | 7,000 |
| Validation | next 3 | 1,500 |
| Test | final 3 | 1,500 |

The split must be preserved. Random row-level splitting would leak future market regimes into model development.

## Horizon eligibility

Not every active order can be observed for every requested horizon. The analysis therefore uses horizon-specific eligibility masks.

| Horizon | Eligible rows | Positive fill rate |
|---:|---:|---:|
| 5 seconds | 9,722 | 10.6% |
| 30 seconds | 8,808 | 41.9% |
| 60 seconds | 8,252 | 63.2% |

Ineligible rows are excluded rather than labelled as non-fills.

## Pre-run integrity checklist

Before modelling, confirm that:

- there are 10,000 unique order identifiers;
- split labels contain 7,000 train, 1,500 validation, and 1,500 test rows;
- each history record contains 50 snapshots strictly before order submission;
- join keys match between the workbook and history JSON;
- eligibility and target columns are consistent for each horizon;
- no feature uses information observed after `t0`;
- only the two canonical source files are present in `data/raw/`.

The first notebook performs the detailed audit and exports its diagnostics to `results/`.

## Data availability statement

The repository contains derived metrics, figures, and documentation so the completed analysis can be reviewed without the raw files. Full reproduction requires separately authorized access to both teaching-data files. The MIT license for the code does not change the ownership or redistribution terms of the source data.
