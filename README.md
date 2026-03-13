# 🏎️ F1 Race Predictor

A machine learning pipeline that predicts **Formula 1 race winners and full finishing order** using real historical data from the [OpenF1](https://openf1.org/) API.

> Built as a portfolio project to explore ML with real-world motorsport data.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://hamhoumf1racepredictor.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

**[👉 hamhoumf1racepredictor.streamlit.app](https://hamhoumf1racepredictor.streamlit.app)**

---

## 🎯 What It Predicts

| Target | Description |
|---|---|
| 🥇 **Race Winner** | The driver most likely to finish 1st |
| 🏆 **Podium (Top 3)** | Which 3 drivers will finish on the podium |
| 📋 **Full Finishing Order** | Ranked prediction for all 20 drivers |

---

## 📊 Model Results

Trained on **2023 + 2024** seasons, tested on the full **2025** season (30 races):

| Metric | Score | Meaning |
|---|---|---|
| 🥇 **Winner Accuracy** | **30.0%** | Correctly predicted the winner in 9/30 races |
| 🏆 **Podium Accuracy** | **56.7%** | Got more than half the podium right on average |
| 📈 **Spearman Rank Corr** | **0.614** | Strong correlation between predicted and actual order |
| 📉 **Avg Position Error** | **3.71** | Average miss of ~3-4 positions per driver |

> **Context:** a random baseline predicts the winner ~5% of the time (1 in 20 drivers).
> This model is **6× better than random** on winner prediction.

### 🏆 Best Race: Jeddah Round 6
- ✅ Winner correct (PIA — McLaren)
- 🏆 2/3 podium correct
- 📈 0.88 rank correlation — near-perfect grid order prediction

---

## 🧠 How It Works

### Pipeline
```
OpenF1 API → raw CSVs → feature engineering → ml_dataset.csv → XGBoost → predictions
```

### Features

| Feature | Why It Matters |
|---|---|
| Driver championship rank | Best proxy for overall pace + skill |
| Avg points (last 3 races) | Captures current momentum |
| Constructor championship rank | Team car performance |
| Rainfall flag | Wet races reshuffle the grid |
| Avg pit stop duration | Strategy efficiency |
| Circuit (encoded) | Some drivers excel at specific tracks |
| Team (encoded) | Constructor pace varies by circuit |

### Model
**XGBoost Regressor** — predicts a finish score per driver per race.
Drivers are ranked by score within each race to produce the full predicted order.

---

## 🗂️ Project Structure

```
f1-race-predictor/
│
├── data/
│   ├── raw/                     # Raw CSVs from OpenF1 API
│   └── processed/               # ML-ready dataset + predictions
│
├── src/
│   ├── data/
│   │   ├── fetch_openf1.py      # OpenF1 API client (rate-limit aware)
│   │   └── preprocess.py        # Feature engineering pipeline
│   ├── models/
│   │   └── train.py             # XGBoost training + evaluation
│   └── dashboard/
│       └── app.py               # Streamlit dashboard
│
├── models/                      # Saved trained models (.pkl)
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

```bash
# 1. Clone & setup
git clone https://github.com/YOUR_USERNAME/f1-race-predictor.git
cd f1-race-predictor
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Fetch data (takes ~20 min, respects API rate limits automatically)
python src/data/fetch_openf1.py

# 3. Preprocess + engineer features
python src/data/preprocess.py

# 4. Train model
python src/models/train.py

# 5. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## 🗺️ Roadmap

- [x] OpenF1 data fetcher with automatic rate-limit handling
- [x] Feature engineering (standings, form, weather, pit stops)
- [x] XGBoost model — winner + full finishing order prediction
- [x] Streamlit dashboard with 4 pages
- [x] Deployed live on Streamlit Cloud
- [ ] Add qualifying position as a feature
- [ ] Circuit-specific model variants
- [ ] Predict upcoming race (live current season data)
- [ ] Docker support

---

## 🔌 API

| API | Cost | Rate Limit |
|---|---|---|
| [OpenF1](https://openf1.org/) | Free & open | 4 req/sec, 500 req/hour |

Rate limits are handled automatically — the fetcher backs off and retries on 429 errors.

---

## 📄 License

MIT — use it, learn from it, build on it.
