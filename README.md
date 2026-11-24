# ⚡ EV Charging Infrastructure Analysis

A full end-to-end data analytics & machine learning project analysing global EV charging infrastructure and predicting high-priority cities for new station investment.

---

## 📁 Project Structure

```
ev-charging-infrastructure-analysis/
│
├── data/                          # Raw CSV datasets (add your files here)
│   ├── charging_station.csv
│   ├── charging_station_ml.csv
│   ├── country_summary.csv
│   ├── world_summary.csv
│   └── ev_models.csv
│
├── notebooks/
│   └── eda.ipynb                  # Full EDA walkthrough notebook
│
├── scripts/
│   ├── data_cleaning.py           # Data loading & cleaning
│   ├── feature_engineering.py     # Feature creation & target variable
│   └── model_training.py          # ML model training & evaluation
│
├── models/
│   └── trained_model.pkl          # Saved best model (auto-generated)
│
├── dashboard/
│   ├── app.py                     # Streamlit dashboard app
│   └── images/charts/             # Saved chart PNGs
│
├── README.md
└── requirements.txt
```

---

## 🗂️ Datasets

| File | Description | Rows |
|------|-------------|------|
| `charging_station.csv` | Individual charging stations globally | 242,417 |
| `charging_station_ml.csv` | City-level aggregated metrics | varies |
| `country_summary.csv` | Country-level fast-charge summaries | varies |
| `world_summary.csv` | Enriched country power & speed metrics | varies |
| `ev_models.csv` | 129 Battery-Electric Vehicle catalogue | 129 |

---

## 🚀 Getting Started

### 1. Clone / Download the project
```bash
cd your-workspace
# place this folder here
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your data files
Place the 5 CSV files inside the `data/` folder.

### 5. Run the scripts in order
```bash
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/model_training.py
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 7. Or open the notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 📊 Key Business Questions Answered

1. Which 10 countries have the most charging stations?
2. What share of stations offer fast/ultra-fast DC charging?
3. How does power capacity (kW) vary across countries?
4. Which EV body styles & manufacturers lead the market?
5. How has the EV model catalogue grown year-on-year?
6. Where are the largest gaps in fast-charging infrastructure?

---

## 🤖 ML Pipeline

- **Target**: Binary classification — High vs Low priority city for new station
- **Models**: Logistic Regression, Random Forest, Gradient Boosting
- **Metric**: ROC-AUC
- **Output**: Composite Priority Score (0–100) per city + recommendation tier

### Recommendation Tiers
| Score | Label |
|-------|-------|
| ≥ 70 | 🔴 Urgent — Install Immediately |
| 50–69 | 🟠 High — Strong Candidate |
| 35–49 | 🟡 Medium — Monitor Growth |
| < 35 | 🟢 Low — Already Well-Served |

---

## 📈 Output Files

After running all scripts, these files are generated:

| File | Description |
|------|-------------|
| `world_enriched.csv` | Country data with readiness scores |
| `ev_models_clean.csv` | Cleaned EV model catalogue |
| `city_ml_clean.csv` | City-level ML-ready features |
| `city_priority_predictions.csv` | Full predictions for all cities |
| `top100_priority_cities.csv` | Top 100 investment targets |
| `model_performance.csv` | AUC scores for all 3 models |
| `dashboard_data.json` | EDA data for dashboard |
| `ml_dashboard_data.json` | ML results for dashboard |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualisation
- **Scikit-learn** — machine learning
- **Streamlit** — interactive dashboard
- **Jupyter** — exploratory notebook

---

## 💡 Key Insights

- 🇺🇸 USA dominates with ~33% of all global stations
- ⚡ Only 5.6% of stations are ultra-fast DC (≥150 kW) — massive upgrade opportunity
- 🚗 SUVs represent 48% of all BEV models in catalogue
- 🇨🇳 China leads EV model origin count (16 models)
- 📈 BEV model launches doubled in 2020–2024 vs. all prior years combined
