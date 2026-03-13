"""
src/data/preprocess.py
======================
Merges all raw OpenF1 CSVs into a single clean ML-ready dataset.

Steps:
  1. Load all raw CSVs
  2. Merge into one table (one row = one driver in one race)
  3. Engineer standing features (championship points, constructor points)
  4. Engineer form features (recent avg finish, recent avg points)
  5. Engineer pit stop features (pit count, avg pit duration)
  6. Engineer weather features (temperature, rainfall flag)
  7. Encode categorical columns
  8. Save to data/processed/ml_dataset.csv

Usage:
    python src/data/preprocess.py
"""

import pandas as pd
import numpy as np
import os
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Step 1: Load raw CSVs ─────────────────────────────────────────────────────

def load_raw_data() -> dict:
    logger.info("📂 Loading raw CSV files...")

    files = {
        "sessions":     "sessions.csv",
        "drivers":      "drivers.csv",
        "results":      "race_results.csv",
        "pit_stops":    "pit_stops.csv",
        "weather":      "weather.csv",
    }

    data = {}
    for name, filename in files.items():
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            logger.warning(f"  Missing file: {path} — skipping")
            data[name] = pd.DataFrame()
        else:
            df = pd.read_csv(path)
            logger.success(f"  Loaded {name:12s} → {len(df):,} rows")
            data[name] = df

    return data


# ── Step 2: Build base table ──────────────────────────────────────────────────

def build_base_table(data: dict) -> pd.DataFrame:
    """
    Core merge: race_results + drivers + sessions
    One row = one driver in one race.
    Only keeps Race sessions (not Qualifying, Practice).
    """
    logger.info("\n🔗 Building base table...")

    results  = data["results"].copy()
    drivers  = data["drivers"].copy()
    sessions = data["sessions"].copy()

    if results.empty or sessions.empty:
        raise ValueError("race_results.csv or sessions.csv is empty — cannot build base table")

    # Keep only Race sessions
    race_sessions = sessions[sessions["session_type"] == "Race"].copy()
    race_sessions = race_sessions[[
        "session_key", "year", "circuit_short_name",
        "country_name", "location", "meeting_key", "date_start"
    ]].drop_duplicates()

    # Parse date so we can sort chronologically
    race_sessions["date_start"] = pd.to_datetime(race_sessions["date_start"], utc=True, errors="coerce")
    race_sessions = race_sessions.sort_values("date_start")

    # Assign a round number per year (1 = first race of the season)
    race_sessions["round"] = race_sessions.groupby("year").cumcount() + 1

    # Merge results with session metadata
    base = results.merge(race_sessions, on="session_key", how="inner")

    # Merge driver info — deduplicate drivers first (keep latest entry per driver/session)
    if not drivers.empty:
        drivers_clean = (
            drivers
            .drop_duplicates(subset=["session_key", "driver_number"])
            [["session_key", "driver_number", "full_name",
              "name_acronym", "team_name", "country_code"]]
        )
        base = base.merge(drivers_clean, on=["session_key", "driver_number"], how="left")

    # Clean position column — convert to numeric, drop rows with no position
    base["position"] = pd.to_numeric(base["position"], errors="coerce")
    base = base.dropna(subset=["position"])
    base["position"] = base["position"].astype(int)

    # Sort by year → round → position for clean ordering
    base = base.sort_values(["year", "round", "position"]).reset_index(drop=True)

    logger.success(f"  Base table: {len(base):,} rows × {base.shape[1]} columns")
    logger.info(f"  Seasons: {sorted(base['year'].unique().tolist())}")
    logger.info(f"  Races:   {base.groupby('year')['round'].max().to_dict()}")

    return base


# ── Step 3: Points & standings ────────────────────────────────────────────────

# Official F1 points system (top 10 finishers)
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
              6: 8,  7: 6,  8: 4,  9: 2,  10: 1}

def add_points_and_standings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - points_scored: F1 points for this race result
      - driver_championship_points: cumulative driver points BEFORE this race
      - constructor_championship_points: cumulative team points BEFORE this race
      - driver_championship_rank: rank among all drivers BEFORE this race
      - constructor_championship_rank: rank among all teams BEFORE this race
    """
    logger.info("\n🏆 Computing championship standings...")

    df = df.copy()
    df["points_scored"] = df["position"].map(POINTS_MAP).fillna(0).astype(int)

    # We compute standings BEFORE each race (so the model sees
    # what the standings looked like going INTO the race)
    driver_cum_points      = {}   # driver_number → cumulative points so far
    constructor_cum_points = {}   # team_name → cumulative points so far

    driver_champ_pts_list      = []
    constructor_champ_pts_list = []

    for _, row in df.sort_values(["year", "round", "position"]).iterrows():
        driver  = row["driver_number"]
        team    = row.get("team_name", "Unknown")
        year    = row["year"]
        round_  = row["round"]

        # Key includes year so standings reset each season
        d_key = (year, driver)
        t_key = (year, team)

        # Record points BEFORE this race (what the model sees as input)
        driver_champ_pts_list.append({
            "session_key":   row["session_key"],
            "driver_number": driver,
            "driver_champ_pts_before":      driver_cum_points.get(d_key, 0),
            "constructor_champ_pts_before": constructor_cum_points.get(t_key, 0),
        })

        # Now accumulate points from this race
        driver_cum_points[d_key]      = driver_cum_points.get(d_key, 0)      + row["points_scored"]
        constructor_cum_points[t_key] = constructor_cum_points.get(t_key, 0) + row["points_scored"]

    standings_df = pd.DataFrame(driver_champ_pts_list)
    df = df.merge(standings_df, on=["session_key", "driver_number"], how="left")

    # Compute championship ranks per race (based on points before race)
    def add_ranks(df, pts_col, rank_col):
        ranks = (
            df.groupby(["year", "round"])[pts_col]
            .rank(ascending=False, method="min")
        )
        df[rank_col] = ranks
        return df

    df = add_ranks(df, "driver_champ_pts_before",      "driver_champ_rank_before")
    df = add_ranks(df, "constructor_champ_pts_before",  "constructor_champ_rank_before")

    logger.success("  ✓ Championship standings added")
    return df


# ── Step 4: Recent form ───────────────────────────────────────────────────────

def add_recent_form(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    For each driver in each race, compute their average finishing
    position and average points over their last N races.
    Uses only past races (no data leakage).
    """
    logger.info(f"\n📈 Computing recent form (last {window} races)...")

    df = df.sort_values(["year", "round", "driver_number"]).copy()

    avg_pos_list    = []
    avg_pts_list    = []
    races_done_list = []

    for _, row in df.iterrows():
        driver = row["driver_number"]
        year   = row["year"]
        round_ = row["round"]

        # Past races for this driver (same year, earlier rounds)
        past = df[
            (df["driver_number"] == driver) &
            (df["year"] == year) &
            (df["round"] < round_)
        ].sort_values("round").tail(window)

        races_done = len(past)
        avg_pos    = past["position"].mean()       if races_done > 0 else np.nan
        avg_pts    = past["points_scored"].mean()  if races_done > 0 else np.nan

        avg_pos_list.append(avg_pos)
        avg_pts_list.append(avg_pts)
        races_done_list.append(races_done)

    df[f"avg_finish_last{window}"]  = avg_pos_list
    df[f"avg_points_last{window}"]  = avg_pts_list
    df["races_completed_this_season"] = races_done_list

    logger.success("  ✓ Recent form features added")
    return df


# ── Step 5: Pit stop features ─────────────────────────────────────────────────

def add_pit_stop_features(df: pd.DataFrame, pit_stops: pd.DataFrame) -> pd.DataFrame:
    """
    Per race per driver:
      - pit_stop_count: how many pit stops they made
      - avg_pit_duration: average pit stop duration (seconds)
    """
    logger.info("\n🔧 Adding pit stop features...")

    if pit_stops.empty:
        logger.warning("  No pit stop data — skipping")
        df["pit_stop_count"]    = np.nan
        df["avg_pit_duration"]  = np.nan
        return df

    pit_stops = pit_stops.copy()
    pit_stops["pit_duration"] = pd.to_numeric(pit_stops["pit_duration"], errors="coerce")

    pit_agg = (
        pit_stops
        .groupby(["session_key", "driver_number"])
        .agg(
            pit_stop_count   = ("lap_number",    "count"),
            avg_pit_duration = ("pit_duration",  "mean"),
        )
        .reset_index()
    )

    df = df.merge(pit_agg, on=["session_key", "driver_number"], how="left")
    df["pit_stop_count"]   = df["pit_stop_count"].fillna(0)

    logger.success("  ✓ Pit stop features added")
    return df


# ── Step 6: Weather features ──────────────────────────────────────────────────

def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Per race session:
      - air_temperature: mean air temp during session
      - track_temperature: mean track temp
      - rainfall_flag: True if it rained
    """
    logger.info("\n🌦️  Adding weather features...")

    if weather.empty:
        logger.warning("  No weather data — skipping")
        return df

    weather_cols = ["session_key"]
    for col in ["air_temperature", "track_temperature", "humidity",
                "wind_speed", "rainfall_flag"]:
        if col in weather.columns:
            weather_cols.append(col)

    weather_clean = weather[weather_cols].drop_duplicates(subset=["session_key"])
    df = df.merge(weather_clean, on="session_key", how="left")

    logger.success("  ✓ Weather features added")
    return df


# ── Step 7: Encode categoricals ───────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string columns to numeric codes so ML models can use them.
    Saves the mapping to a CSV so you can decode predictions later.
    """
    logger.info("\n🔢 Encoding categorical columns...")

    cat_cols = ["circuit_short_name", "team_name", "name_acronym", "country_code"]
    mappings = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Unknown")
        codes   = pd.Categorical(df[col])
        mapping = dict(enumerate(codes.categories))
        mappings[col] = mapping

        df[f"{col}_encoded"] = codes.codes

    # Save mappings so we can decode predictions later
    mapping_rows = []
    for col, mapping in mappings.items():
        for code, label in mapping.items():
            mapping_rows.append({"column": col, "code": code, "label": label})

    if mapping_rows:
        mapping_df = pd.DataFrame(mapping_rows)
        mapping_path = os.path.join(PROCESSED_DIR, "encoding_map.csv")
        mapping_df.to_csv(mapping_path, index=False)
        logger.success(f"  Saved encoding map → {mapping_path}")

    logger.success("  ✓ Categorical encoding done")
    return df


# ── Step 8: Final cleanup & save ─────────────────────────────────────────────

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup:
      - Fill remaining NaNs with sensible defaults
      - Define and reorder the final feature columns
      - Save to data/processed/ml_dataset.csv
    """
    logger.info("\n🧹 Finalizing dataset...")

    # Fill NaN in form features with median (for first race of season)
    form_cols = [c for c in df.columns if "avg_finish" in c or "avg_points" in c]
    for col in form_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill remaining numeric NaNs with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Define the final column order
    id_cols = [
        "session_key", "year", "round", "circuit_short_name",
        "country_name", "driver_number", "name_acronym",
        "full_name", "team_name", "date_start"
    ]

    feature_cols = [
        # Championship context
        "driver_champ_pts_before",
        "driver_champ_rank_before",
        "constructor_champ_pts_before",
        "constructor_champ_rank_before",
        # Recent form
        "avg_finish_last3",
        "avg_points_last3",
        "races_completed_this_season",
        # Pit stops
        "pit_stop_count",
        "avg_pit_duration",
        # Weather
        "air_temperature",
        "track_temperature",
        "rainfall_flag",
        # Encoded categoricals
        "circuit_short_name_encoded",
        "team_name_encoded",
    ]

    target_col = ["position"]   # What we're predicting

    # Only keep columns that actually exist
    id_cols      = [c for c in id_cols      if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    final_cols = id_cols + feature_cols + target_col
    df = df[final_cols]

    # Save
    out_path = os.path.join(PROCESSED_DIR, "ml_dataset.csv")
    df.to_csv(out_path, index=False)

    logger.success(f"\n✅ Saved ML dataset → {out_path}")
    logger.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"   Features: {feature_cols}")

    return df


# ── Master pipeline ───────────────────────────────────────────────────────────

def run_pipeline():
    logger.info("🏎️  Starting preprocessing pipeline...\n")

    # 1. Load
    data = load_raw_data()

    # 2. Base table
    base = build_base_table(data)

    # 3. Points & standings
    base = add_points_and_standings(base)

    # 4. Recent form
    base = add_recent_form(base, window=3)

    # 5. Pit stops
    base = add_pit_stop_features(base, data["pit_stops"])

    # 6. Weather
    base = add_weather_features(base, data["weather"])

    # 7. Encode categoricals
    base = encode_categoricals(base)

    # 8. Finalize & save
    final = finalize(base)

    # Print a preview
    print("\n📊 Dataset Preview:")
    print(final.head(10).to_string(index=False))

    print(f"\n📐 Shape: {final.shape[0]:,} rows × {final.shape[1]} columns")
    print(f"\n🎯 Target distribution (finishing positions):")
    print(final["position"].value_counts().sort_index().to_string())

    return final


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_pipeline()
