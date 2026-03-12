# 🏎️ F1 Race Predictor

A machine learning pipeline that predicts **Formula 1 race winners and full finishing order** using real telemetry and historical data from the [OpenF1](https://openf1.org/) and [FastF1](https://docs.fastf1.dev/) APIs.

> Built as a portfolio project to explore ML with real-world motorsport data.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Demo

> *(Screenshot of Streamlit dashboard — coming soon)*

---

## 🎯 What It Predicts

| Target | Description |
|---|---|
| 🥇 **Race Winner** | The driver most likely to finish 1st |
| 🏆 **Podium (Top 3)** | Probability each driver finishes on the podium |
| 📋 **Full Finishing Order** | Ranked list of all drivers for a given race |

---

## 🗂️ Project Structure

```
f1-race-predictor/
│
├── data/                        # Raw & processed data (git-ignored)
│   ├── raw/
│   └── processed/
│
├── notebooks/                   # Exploration & experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data/
│   │   ├── fetch_openf1.py      # OpenF1 API client
│   │   ├── fetch_fastf1.py      # FastF1 data loader
│   │   └── preprocess.py        # Cleaning & merging
│   │
│   ├── features/
│   │   └── engineer.py          # Feature creation logic
│   │
│   ├── models/
│   │   ├── train.py             # Model training
│   │   ├── predict.py           # Run predictions
│   │   └── evaluate.py          # Metrics & scoring
│   │
│   └── dashboard/
│       └── app.py               # Streamlit dashboard
│
├── models/                      # Saved trained models (.pkl)
├── requirements.txt
├── .env.example                 # Environment variable template
├── run_pipeline.py              # One command to run everything
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/f1-race-predictor.git
cd f1-race-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# No API keys needed — OpenF1 and FastF1 are both free & open!
```

---

## 🚀 Usage

### Run the full pipeline (fetch → features → train → predict)
```bash
python run_pipeline.py
```

### Launch the Streamlit dashboard
```bash
streamlit run src/dashboard/app.py
```

### Explore the notebooks
```bash
jupyter notebook notebooks/
```

---

## 🧠 ML Approach

### Model
We use **XGBoost** with a ranking objective — for each race, the model scores every driver and ranks them by predicted finish position.

### Features (Phase 1)
| Feature | Source |
|---|---|
| Qualifying position | FastF1 |
| Gap to pole (seconds) | FastF1 |
| Driver championship standing | FastF1 |
| Constructor championship standing | FastF1 |
| Circuit (encoded) | FastF1 |

### Features (Phase 2 — planned)
| Feature | Source |
|---|---|
| Driver recent form (avg finish, last 3 races) | FastF1 |
| Circuit-specific historical performance | FastF1 |
| Pit stop count & strategy | OpenF1 |
| Weather (rainfall, temperature) | OpenF1 |
| Lap time consistency (std dev) | FastF1 |

### Evaluation Metrics
- **Top-1 Accuracy** — Did we predict the winner correctly?
- **Top-3 Accuracy** — Did we predict all 3 podium finishers?
- **Spearman Rank Correlation** — How well does our full order match reality?

---

## 📊 Results

> *(Model performance metrics will be added after training)*

| Metric | Score |
|---|---|
| Winner accuracy | TBD |
| Podium accuracy | TBD |
| Rank correlation | TBD |

---

## 🗺️ Roadmap

- [x] Project structure & setup
- [ ] OpenF1 data fetcher (`fetch_openf1.py`)
- [ ] FastF1 data loader (`fetch_fastf1.py`)
- [ ] Data exploration notebook
- [ ] Feature engineering pipeline
- [ ] XGBoost model training
- [ ] Model evaluation & backtesting
- [ ] Streamlit dashboard
- [ ] Docker support

---

## 🔌 APIs Used

| API | Docs | Cost |
|---|---|---|
| [OpenF1](https://openf1.org/) | [openf1.org/docs](https://openf1.org/) | Free |
| [FastF1](https://docs.fastf1.dev/) | [docs.fastf1.dev](https://docs.fastf1.dev/) | Free |

---

## 📚 Learning Resources

- [FastF1 Getting Started](https://docs.fastf1.dev/tutorial.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 🤝 Contributing

Pull requests welcome! If you're also learning F1 + ML, feel free to fork this and experiment.

---

## 📄 License

MIT — use it, learn from it, improve it.
