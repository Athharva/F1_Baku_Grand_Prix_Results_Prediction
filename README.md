# F1_Baku_Grand_Prix_Results_Prediction

# 🏁 F1 Race Outcome Predictor — Azerbaijan Grand Prix 2026

A machine learning pipeline that predicts Formula 1 race winners using historical race results and rolling performance features. Built as a pre-qualifying model for the 2026 Azerbaijan Grand Prix using the FastF1 API and XGBoost.

---

## Problem Statement

Predicting F1 race outcomes is a high-variance classification problem — results are influenced by qualifying position, tyre strategy, weather, mechanical failures, and driver form. This project scopes the problem to **pre-qualifying prediction**, using only historical race results to rank the field by win likelihood before race weekend begins.

---

## Approach

The pipeline follows 6 stages:

```
Race Results → Feature Engineering → Model Training → Evaluation → Baku Prediction → Export
```

1. **Data ingestion** — pulls race results from 2018–2026 via the Ergast API through FastF1, with date-based caching to avoid redundant API calls
2. **Feature engineering** — constructs 9 rolling window features per driver per race
3. **Temporal train-test split** — trains on 2019–2024, tests on 2025 to preserve race integrity and prevent data leakage
4. **Model training** — XGBoost binary classifier trained to predict race winners
5. **Evaluation** — measures top-pick accuracy across the 2025 season
6. **Baku forecast** — generates a normalized win probability ranking for the full 2026 Azerbaijan GP field

---

## Feature Set

| Group | Feature | Description |
|---|---|---|
| Driver Form | `driver_finish_5` | Average finishing position over last 5 races |
| Driver Form | `driver_points_5` | Average points scored over last 5 races |
| Driver Form | `driver_win_rate_5` | Win rate over last 5 races |
| Driver Form | `driver_podium_rate_5` | Podium rate over last 5 races |
| Team Form | `team_points_5` | Average combined team points over last 5 races |
| Team Form | `team_win_rate_5` | Team win rate over last 5 races |
| Circuit History | `circuit_finish_3` | Driver's average finish at this circuit over last 3 visits |
| Circuit History | `circuit_win_rate_3` | Driver's win rate at this circuit over last 3 visits |
| Availability | `driver_history_count` | Number of prior starts available, capped at 5 |

Rolling windows use available history when fewer starts exist; missing values are propagated as `NaN` and handled natively by XGBoost.

---

## Model

- **Algorithm:** XGBoost Binary Classifier (`binary:logistic`)
- **Hyperparameters:** `n_estimators=120`, `max_depth=3`, `learning_rate=0.06`, `reg_lambda=5.0`, `min_child_weight=5`
- **Evaluation metric:** Log loss
- **Train set:** 2019–2024 race results
- **Test set:** 2025 season (held out entirely; no race-level mixing)
- **Prediction:** Normalized win probability scores across the full Baku entry list

---

## Results

- The model correctly predicted several race winners during 2025 season evaluation
- The top pick for the 2026 Azerbaijan Grand Prix was **incorrect**, though sub-predictions (top-5 rankings) showed partial accuracy
- Limitations: the model has no access to qualifying data, weather, tyre strategy, or live lap telemetry — all of which are significant predictive signals in F1

---

## Output Files

| File | Description |
|---|---|
| `baku_predictions.csv` | Full ranked field with normalized win scores |
| `baku_predictions.json` | Predictions with model metadata |
| `baku_inputs.json` | Feature inputs used for the Baku forecast |
| `baku_model.joblib` | Serialized model for inference |
| `baku_model.json` | XGBoost native model format |
| `baku_model_metadata.json` | Feature list and run metadata |
| `Baku_XGBoost_Dashboard_Data.json` | Full dashboard kit including verification cases |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| FastF1 ≥ 3.6 | F1 race data via Ergast API |
| XGBoost ≥ 3.0 | Binary classification model |
| Scikit-learn ≥ 1.6 | Evaluation utilities |
| Pandas / NumPy | Data wrangling and feature engineering |
| Matplotlib | Win probability visualization |
| Joblib | Model serialization |

---

## Setup & Usage

### Prerequisites
```bash
pip install "fastf1>=3.6,<4" "scikit-learn>=1.6,<2" "xgboost>=3.0,<4" matplotlib joblib
```

### Run
Open `F1_Project.ipynb` in Jupyter or Google Colab and run all cells sequentially. The notebook handles caching, feature construction, training, evaluation, and prediction automatically.

To target a different race or season, update:
```python
TARGET_YEAR = 2026       # Forecast year
START_YEAR = 2018        # Historical data start
```

To supply a custom entry list for the target race, place a `baku_entries.csv` file in the working directory with columns: `driverId`, `constructorId`, `driver`, `team`.

---

## Limitations

- **Pre-qualifying only** — no qualifying position, grid penalties, or live telemetry
- **No weather or tyre data** — significant omission given Baku's street circuit characteristics
- **Binary target** — models win/no-win only; does not predict podium or points finishes
- **Small circuit history** — some drivers have fewer than 3 Baku starts, leading to sparse circuit features

---

## Future Improvements

- Incorporate qualifying lap times and grid positions as features
- Add tyre compound and strategy data via FastF1 telemetry
- Extend target variable to top-3 or top-5 (ordinal classification)
- Build a Streamlit dashboard for interactive race-weekend predictions

---

## License

MIT License
