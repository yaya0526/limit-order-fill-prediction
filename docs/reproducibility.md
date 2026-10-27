# Reproducibility guide

The repository supports two levels of verification:

1. **Raw-data-free checks** validate the published code, notebooks, links, and derived result contracts.
2. **Full notebook reproduction** additionally requires the two restricted teaching-data files described in [`data/README.md`](../data/README.md).

## Supported environment

- Python 3.11 is the reference version used by CI.
- Runtime dependencies use compatible version ranges in `requirements.txt`.
- Development and verification dependencies are collected in `requirements-dev.txt`.
- `environment.yml` provides an equivalent Conda entry point.

## Virtual environment setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Verify the published repository

These checks do not require the raw Excel or JSON inputs:

```bash
python scripts/check_repository.py
python -m pytest -q
```

On systems with `make`, the equivalent single command is:

```bash
make ci
```

The validation script checks:

- the expected eight-notebook sequence;
- JSON readability and the final-metric schema;
- confusion-matrix and probability-range consistency;
- local Markdown links;
- required result and policy artifacts;
- R1–R4 acceptance records, including independent-simulation coverage and aggregate-only outputs;
- accidental personal absolute paths.

## Re-run the notebooks

1. Place the canonical Excel and compressed JSON files in `data/raw/`.
2. Start Jupyter from the repository root with `jupyter lab`.
3. Run notebooks `01` through `08` in ascending order.

The notebooks resolve paths from the repository root. Advanced users can override output locations with the environment variables already documented in the evaluation notebooks, including `LIMIT_ORDER_PROJECT_ROOT`, `LIMIT_ORDER_ARTIFACT_DIR`, and `LIMIT_ORDER_FIGURE_DIR`.

The scripted R1–R4 extensions use the workbook but not the history JSON. Run the latest independent simulation study with:

```bash
python scripts/run_independent_simulation.py --excel data/raw/AI_Order_Execution.xlsx
```

The simulator itself does not read the workbook; the workbook is used only to reconstruct the frozen source estimator and calibrator. Only aggregate scenario metrics are committed.

## What CI does not do

CI deliberately does not execute the full notebook pipeline because the source teaching data are not redistributed. It verifies the reusable source code, deterministic unit tests, committed notebook structure, and published result contracts instead.
