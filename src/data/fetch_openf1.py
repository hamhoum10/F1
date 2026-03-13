"""
src/data/fetch_openf1.py
========================
Fetches F1 data from the OpenF1 REST API (https://openf1.org/)
Covers: sessions, race results, pit stops, weather

Usage:
    python src/data/fetch_openf1.py
"""

import requests
import pandas as pd
import time
import os
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = "https://api.openf1.org/v1"
SEASONS  = [2023, 2024, 2025 , 2026]
RAW_DIR  = "data/raw"

os.makedirs(RAW_DIR, exist_ok=True)


# ── Core API helper ───────────────────────────────────────────────────────────

def fetch(endpoint: str, params: dict = {}, _retry: int = 0) -> list[dict]:
    """
    Make a GET request to the OpenF1 API and return the JSON result.

    Retry strategy:
      - 429 Too Many Requests → wait and retry up to 6 times
          with exponential backoff: 15s, 30s, 60s, 120s, 240s, 300s
      - 404 Not Found         → skip silently (data doesn't exist)
      - Connection error      → retry up to 3 times with 10s wait
      - Any other error       → log warning and return []
    """
    MAX_RETRIES = 6
    BACKOFF     = [15, 30, 60, 120, 240, 300]  # seconds per retry attempt

    url = f"{BASE_URL}/{endpoint}"

    try:
        response = requests.get(url, params=params, timeout=30)

        # ── 429: Rate limited ──────────────────────────────────────────
        if response.status_code == 429:
            if _retry >= MAX_RETRIES:
                logger.error(f"  ✗ Gave up after {MAX_RETRIES} retries: {endpoint} | params={params}")
                return []

            wait = BACKOFF[_retry]

            # Respect Retry-After header if the API sends one
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(wait, int(retry_after))
                except ValueError:
                    pass

            logger.warning(
                f"  ⏳ Rate limited (429) — waiting {wait}s then retrying "
                f"[attempt {_retry + 1}/{MAX_RETRIES}] | {endpoint}"
            )
            time.sleep(wait)
            return fetch(endpoint, params, _retry=_retry + 1)

        # ── 404: No data for this session ──────────────────────────────
        if response.status_code == 404:
            logger.debug(f"  No data (404): {endpoint} | params={params}")
            return []

        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        if _retry >= 3:
            logger.warning(f"  ✗ Connection failed after 3 retries: {endpoint}")
            return []
        wait = 10 * (_retry + 1)
        logger.warning(f"  Connection error — retrying in {wait}s [{_retry + 1}/3]")
        time.sleep(wait)
        return fetch(endpoint, params, _retry=_retry + 1)

    except requests.exceptions.HTTPError as e:
        logger.warning(f"  HTTP error: {endpoint} | {e}")
        return []

    except requests.exceptions.RequestException as e:
        logger.warning(f"  Request failed: {endpoint} | {e}")
        return []


# ── Fetchers ──────────────────────────────────────────────────────────────────

def get_sessions(year: int) -> pd.DataFrame:
    """
    Get all sessions (Race, Qualifying, Practice) for a given year.
    We mainly care about 'Race' and 'Qualifying' session types.
    """
    logger.info(f"Fetching sessions for {year}...")
    data = fetch("sessions", {"year": year})

    if not data:
        logger.warning(f"No sessions found for {year}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Keep only useful columns
    cols = ["session_key", "session_name", "session_type",
            "date_start", "year", "circuit_key", "circuit_short_name",
            "country_name", "location", "meeting_key"]
    cols = [c for c in cols if c in df.columns]

    return df[cols]


def get_race_results(session_key: int) -> pd.DataFrame:
    """
    Get the final position of each driver for a race session.
    OpenF1 'position' endpoint gives real-time positions —
    we take the LAST recorded position per driver as final result.
    """
    logger.info(f"  Fetching race results for session {session_key}...")
    data = fetch("position", {"session_key": session_key})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Take last known position per driver = final finishing position
    df = df.sort_values("date")
    final_positions = df.groupby("driver_number").last().reset_index()
    final_positions["session_key"] = session_key

    cols = ["session_key", "driver_number", "position"]
    cols = [c for c in cols if c in final_positions.columns]

    return final_positions[cols]


def get_drivers(session_key: int) -> pd.DataFrame:
    """
    Get driver information (name, team, abbreviation) for a session.
    """
    logger.info(f"  Fetching drivers for session {session_key}...")
    data = fetch("drivers", {"session_key": session_key})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    cols = ["session_key", "driver_number", "full_name",
            "name_acronym", "team_name", "team_colour", "country_code"]
    cols = [c for c in cols if c in df.columns]

    return df[cols]


def get_pit_stops(session_key: int) -> pd.DataFrame:
    """
    Get pit stop data for a race session.
    Returns each pit stop with lap number and duration.
    """
    logger.info(f"  Fetching pit stops for session {session_key}...")
    data = fetch("pit", {"session_key": session_key})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["session_key"] = session_key

    cols = ["session_key", "driver_number", "lap_number",
            "pit_duration", "date"]
    cols = [c for c in cols if c in df.columns]

    return df[cols]


def get_weather(session_key: int) -> pd.DataFrame:
    """
    Get weather snapshots during a session.
    We'll aggregate these into a single row per session (mean values).
    """
    logger.info(f"  Fetching weather for session {session_key}...")
    data = fetch("weather", {"session_key": session_key})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Aggregate into one row per session
    numeric_cols = ["air_temperature", "track_temperature",
                    "humidity", "wind_speed", "rainfall"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    agg = df[numeric_cols].mean().to_frame().T
    agg["session_key"] = session_key
    agg["rainfall_flag"] = (df["rainfall"].sum() > 0) if "rainfall" in df.columns else False

    return agg


# ── Master pipeline ───────────────────────────────────────────────────────────

def fetch_all_seasons(seasons: list[int] = SEASONS):
    """
    Main function: loops over all seasons and saves data to CSV files.
    """
    all_sessions     = []
    all_results      = []
    all_drivers      = []
    all_pit_stops    = []
    all_weather      = []

    for year in seasons:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing season {year}")
        logger.info(f"{'='*50}")

        # 1. Get all sessions for this year
        sessions_df = get_sessions(year)
        if sessions_df.empty:
            continue

        all_sessions.append(sessions_df)

        # 2. Filter to Race and Qualifying sessions only
        target_types = ["Race", "Qualifying"]
        filtered = sessions_df[sessions_df["session_type"].isin(target_types)]

        for _, session in filtered.iterrows():
            session_key  = int(session["session_key"])
            session_type = session["session_type"]
            circuit      = session.get("circuit_short_name", "Unknown")

            logger.info(f"\n→ {year} | {circuit} | {session_type} (key={session_key})")

            # 3. Fetch data for this session
            drivers   = get_drivers(session_key)
            all_drivers.append(drivers)

            if session_type == "Race":
                results   = get_race_results(session_key)
                pit_stops = get_pit_stops(session_key)
                weather   = get_weather(session_key)

                all_results.append(results)
                all_pit_stops.append(pit_stops)
                all_weather.append(weather)

            # Be polite to the API — small delay between requests
            time.sleep(1.2)

    # 4. Combine and save everything
    logger.info("\n💾 Saving data to CSV...")

    def save(dfs: list, filename: str):
        if dfs:
            combined = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
            path = os.path.join(RAW_DIR, filename)
            combined.to_csv(path, index=False)
            logger.success(f"  Saved {len(combined)} rows → {path}")
            return combined
        return pd.DataFrame()

    sessions_combined  = save(all_sessions,  "sessions.csv")
    drivers_combined   = save(all_drivers,   "drivers.csv")
    results_combined   = save(all_results,   "race_results.csv")
    pitstops_combined  = save(all_pit_stops, "pit_stops.csv")
    weather_combined   = save(all_weather,   "weather.csv")

    logger.success("\n✅ OpenF1 data fetch complete!")

    return {
        "sessions":   sessions_combined,
        "drivers":    drivers_combined,
        "results":    results_combined,
        "pit_stops":  pitstops_combined,
        "weather":    weather_combined,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = fetch_all_seasons(SEASONS)

    # Quick summary
    print("\n📊 Data Summary:")
    for name, df in data.items():
        if not df.empty:
            print(f"  {name:12s} → {len(df):,} rows, {df.shape[1]} columns")