# VinUni Datathon 2026 - Round 1

This repository contains our Round 1 submission for Datathon 2026 - The Gridbreaker. It includes the source code, notebooks, generated figures, report, and final forecasting submission.

## Repository Structure

```text
.
├── data/                  # Local raw CSV files, ignored by git
├── notebooks/             # Part 1 and baseline notebooks
├── reports/               # Submitted reports and problem statement
│   └── figures/           # Generated EDA and forecasting figures
├── src/
│   ├── part_2/            # Visualization and analysis scripts
│   └── part_3/            # Sales forecasting model
├── submission.csv         # Final Kaggle submission
├── requirements.txt
└── README.md
```

## Data

Raw CSV files are not committed to this repository. Download the competition data from Google Drive:

https://drive.google.com/drive/folders/1G8TC9KNA4Ar2LIzl9cpn7CuWsp6JA2Vo?usp=sharing

Place the CSV files directly under `data/` before running the code, for example:

```text
data/sales.csv
data/sample_submission.csv
data/orders.csv
data/order_items.csv
data/promotions.csv
data/web_traffic.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Results

Run the final forecasting model:

```bash
python src/part_3/model_34.py
```

This writes the final submission format to `submission.csv`.

Regenerate Part 2 figures by running the scripts in `src/part_2/` from the repository root. Figures are written to `reports/figures/`.

## Submission Artifacts

- `submission.csv`: final Kaggle submission file.
- `reports/Datathon_report.pdf`: final report.
- `reports/part_2.pdf`: Part 2 visualization and analysis material.

