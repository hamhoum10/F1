"""
src/models/train.py
===================
Trains an XGBoost model to predict F1 race finishing positions.

Strategy:
  - Train on 2023 + 2024 seasons
  - Test on 2025 season (unseen data)
  - Model predicts a "finish score" per driver — lower = better finish
  - We rank drivers by score to get predicted finishing order

Usage:
    python src/models/train.py
"""

import pandas as pd
import numpy as np
import os
import joblib
from loguru import logger
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_YEARS = [2023, 2024]
TEST_YEARS  = [2025]

# Features the model will use
FEATURE_COLS = [
    "driver_champ_pts_before",
    "driver_champ_rank_before",
    "constructor_champ_pts_before",
    "constructor_champ_rank_before",
    "avg_finish_last3",
    "avg_points_last3",
    "races_completed_this_season",
    "pit_stop_count",
    "avg_pit_duration",
    "air_temperature",
    "track_temperature",
    "rainfall_flag",
    "circuit_short_name_encoded",
    "team_name_encoded",
]

TARGET_COL = "position"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "ml_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Run preprocess.py first.")

    df = pd.read_csv(path)
    logger.info(f"📂 Loaded dataset: {len(df):,} rows × {df.shape[1]} columns")
    logger.info(f"   Seasons: {sorted(df['year'].unique().tolist())}")
    logger.info(f"   Races per season: {df.groupby('year')['round'].nunique().to_dict()}")
    return df


# ── Train / test split ────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    Temporal split — never train on future data.
    Train: 2023 + 2024  |  Test: 2025
    """
    # Only keep features that exist in the dataset
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        logger.warning(f"  Missing features (will skip): {missing}")

    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    test_df  = df[df["year"].isin(TEST_YEARS)].copy()

    logger.info(f"\n📊 Split summary:")
    logger.info(f"   Train: {len(train_df):,} rows ({TRAIN_YEARS})")
    logger.info(f"   Test:  {len(test_df):,} rows  ({TEST_YEARS})")

    X_train = train_df[available_features]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[available_features]
    y_test  = test_df[TARGET_COL]

    return X_train, y_train, X_test, y_test, test_df, available_features


# ── Train model ───────────────────────────────────────────────────────────────

def train_model(X_train, y_train):
    """
    Train an XGBoost regressor to predict finishing position.
    Lower predicted score = better predicted finish.
    """
    logger.info("\n🤖 Training XGBoost model...")

    # Scale features — helps XGBoost converge better
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = XGBRegressor(
        n_estimators      = 300,
        max_depth         = 4,       # shallow trees avoid overfitting on small data
        learning_rate     = 0.05,
        subsample         = 0.8,     # use 80% of rows per tree
        colsample_bytree  = 0.8,     # use 80% of features per tree
        min_child_weight  = 3,
        reg_alpha         = 0.1,     # L1 regularisation
        reg_lambda        = 1.0,     # L2 regularisation
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )

    model.fit(X_scaled, y_train)

    logger.success("  ✓ Model trained")
    return model, scaler


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, scaler, X_test, y_test, test_df, feature_cols):
    """
    Evaluate on 2025 races using three metrics:
      1. Winner accuracy    — did we predict P1 correctly?
      2. Podium accuracy    — did we get all top-3 right?
      3. Spearman rank corr — how well does the full order match?
    """
    logger.info("\n📏 Evaluating on 2025 races...")

    X_scaled  = scaler.transform(X_test)
    predicted = model.predict(X_scaled)

    test_df = test_df.copy()
    test_df["predicted_score"] = predicted

    # Rank drivers within each race by predicted score (lower = better)
    test_df["predicted_position"] = (
        test_df.groupby(["year", "round"])["predicted_score"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    # ── Per-race metrics ──────────────────────────────────────────────────────
    winner_correct  = []
    podium_correct  = []
    spearman_scores = []
    mae_scores      = []

    races = test_df.groupby(["year", "round"])

    for (year, round_), race in races:
        race = race.sort_values("position")

        actual_winner    = race[race["position"] == 1]["driver_number"].values
        predicted_winner = race[race["predicted_position"] == 1]["driver_number"].values

        actual_podium    = set(race[race["position"]    <= 3]["driver_number"].tolist())
        predicted_podium = set(race[race["predicted_position"] <= 3]["driver_number"].tolist())

        winner_correct.append(
            len(actual_winner) > 0 and len(predicted_winner) > 0
            and actual_winner[0] == predicted_winner[0]
        )

        # Podium accuracy: what % of the top 3 did we get right?
        podium_overlap = len(actual_podium & predicted_podium) / 3
        podium_correct.append(podium_overlap)

        # Spearman rank correlation for full grid
        merged = race[["driver_number", "position", "predicted_position"]].dropna()
        if len(merged) > 2:
            corr, _ = spearmanr(merged["position"], merged["predicted_position"])
            spearman_scores.append(corr)

        mae_scores.append(mean_absolute_error(race["position"], race["predicted_position"]))

    # ── Summary ───────────────────────────────────────────────────────────────
    winner_acc  = np.mean(winner_correct)
    podium_acc  = np.mean(podium_correct)
    avg_spearman = np.mean(spearman_scores)
    avg_mae     = np.mean(mae_scores)

    logger.info(f"\n{'='*45}")
    logger.info(f"  📊 Model Evaluation Results (2025 season)")
    logger.info(f"{'='*45}")
    logger.info(f"  🥇 Winner accuracy:         {winner_acc:.1%}  ({sum(winner_correct)}/{len(winner_correct)} races)")
    logger.info(f"  🏆 Podium accuracy:         {podium_acc:.1%}  (avg overlap with actual top 3)")
    logger.info(f"  📈 Spearman rank corr:      {avg_spearman:.3f} (1.0 = perfect order)")
    logger.info(f"  📉 Mean abs position error: {avg_mae:.2f} positions")
    logger.info(f"{'='*45}")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info(f"\n🔍 Top feature importances:")
    for _, row in importance.head(8).iterrows():
        bar = "█" * int(row["importance"] * 100)
        logger.info(f"  {row['feature']:40s} {bar} {row['importance']:.3f}")

    # Save importance
    imp_path = os.path.join(PROCESSED_DIR, "feature_importance.csv")
    importance.to_csv(imp_path, index=False)

    # Save predictions
    pred_path = os.path.join(PROCESSED_DIR, "test_predictions.csv")
    test_df.to_csv(pred_path, index=False)
    logger.success(f"\n  Saved predictions  → {pred_path}")
    logger.success(f"  Saved importances  → {imp_path}")

    return {
        "winner_accuracy":   winner_acc,
        "podium_accuracy":   podium_acc,
        "spearman":          avg_spearman,
        "mae":               avg_mae,
        "predictions":       test_df,
        "feature_importance": importance,
    }


# ── Save model ────────────────────────────────────────────────────────────────

def save_model(model, scaler, feature_cols):
    model_path  = os.path.join(MODELS_DIR, "xgb_f1_predictor.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    feat_path   = os.path.join(MODELS_DIR, "feature_cols.pkl")

    joblib.dump(model,        model_path)
    joblib.dump(scaler,       scaler_path)
    joblib.dump(feature_cols, feat_path)

    logger.success(f"\n💾 Saved model   → {model_path}")
    logger.success(f"   Saved scaler  → {scaler_path}")
    logger.success(f"   Saved features → {feat_path}")


# ── Sample prediction ─────────────────────────────────────────────────────────

def show_sample_prediction(results: dict):
    """Print a nicely formatted prediction for one race."""
    preds = results["predictions"]

    # Pick the last race in the test set
    last_race = preds[preds["round"] == preds["round"].max()]
    last_race = last_race[preds["year"] == preds["year"].max()]
    last_race = last_race.sort_values("predicted_position")

    circuit = last_race["circuit_short_name"].iloc[0] if "circuit_short_name" in last_race.columns else "Unknown"
    year    = last_race["year"].iloc[0]
    round_  = last_race["round"].iloc[0]

    print(f"\n{'='*50}")
    print(f"  🏁 Sample: {year} Round {round_} — {circuit}")
    print(f"{'='*50}")
    print(f"  {'Pred':>5}  {'Actual':>6}  {'Driver':<15}  {'Team'}")
    print(f"  {'-'*45}")

    for _, row in last_race.iterrows():
        name  = row.get("name_acronym", row.get("driver_number", "???"))
        team  = row.get("team_name", "")[:20] if "team_name" in last_race.columns else ""
        pred  = int(row["predicted_position"])
        actual = int(row["position"])
        flag  = "✓" if pred == actual else ("~" if abs(pred - actual) <= 2 else " ")
        print(f"  {pred:>5}  {actual:>6}  {str(name):<15}  {team}  {flag}")

    print(f"{'='*50}\n")


# ── Master pipeline ───────────────────────────────────────────────────────────

def run():
    logger.info("🏎️  F1 Race Predictor — Model Training\n")

    df = load_dataset()
    X_train, y_train, X_test, y_test, test_df, feature_cols = split_data(df)
    model, scaler = train_model(X_train, y_train)
    results = evaluate(model, scaler, X_test, y_test, test_df, feature_cols)
    save_model(model, scaler, feature_cols)
    show_sample_prediction(results)

    logger.success("✅ Training pipeline complete!")
    return model, scaler, results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
