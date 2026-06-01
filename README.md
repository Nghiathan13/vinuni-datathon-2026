# VinUni Datathon 2026 - Round 1

This repository contains the Round 1 submission for **Datathon 2026 - The Gridbreaker**, focused on business analytics and sales forecasting for a simulated Vietnamese fashion e-commerce company.

The main deliverables are the final report, generated visualizations, forecasting source code, and Kaggle submission file.

## Key Visualizations

### Revenue Trend

![Annual revenue by year](reports/figures/annual_revenue.png)

Annual revenue shows the long-term growth trajectory of the business from 2013 to 2022. This view gives a high-level baseline for business scale, helps identify whether growth is stable or volatile, and provides context for later forecasting assumptions.

### Customer Growth and Retention

![Customer growth and retention](reports/figures/merge_2_combined.png)

This combined view compares active and new customer growth with cohort retention behavior. It highlights whether business expansion is driven mainly by new acquisition or repeat purchasing, which is important for evaluating customer loyalty and sustainable revenue growth.

### Gross Profit Forecast

![Gross profit forecast](reports/figures/forecast_gross_profit_2023_2024.png)

The gross profit forecast extends historical profitability into 2023 and 2024. This chart connects the forecasting model to business planning by showing expected profit movement, not only top-line revenue.

More generated figures are available in `reports/figures/`, and the full analysis is summarized in `reports/Datathon_report.pdf`.

## Data

Raw competition CSV files are not committed to this repository. Download the data from Google Drive and place the CSV files directly under `data/`:

https://drive.google.com/drive/folders/1G8TC9KNA4Ar2LIzl9cpn7CuWsp6JA2Vo?usp=sharing

## Run the Forecast

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the final forecasting pipeline:

```bash
python src/part_3/model_34.py
```

The script trains the model from the local `data/` files and writes the Kaggle submission format to `submission.csv`.

## Main Artifacts

- `reports/Datathon_report.pdf` - final written report.
- `submission.csv` - final Kaggle submission file.
- `src/part_3/model_34.py` - final forecasting model.
- `reports/figures/` - generated visualizations used in the analysis.
