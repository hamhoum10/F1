"""
src/dashboard/app.py
====================
Streamlit dashboard for the F1 Race Predictor.

Pages:
  1. 🏁 Race Predictions  — pick any 2025 race, see predicted vs actual order
  2. 📊 Model Performance — winner/podium accuracy, Spearman per race
  3. 🔍 Feature Importance — what the model relies on most
  4. 📈 Driver Form        — recent form trends per driver

Usage:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "F1 Race Predictor",
    page_icon   = "🏎️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #e10600;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e10600;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .race-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        border-left: 4px solid #e10600;
        padding-left: 12px;
        margin-bottom: 16px;
    }
    div[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions():
    path = "data/processed/test_predictions.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_full_dataset():
    path = "data/processed/ml_dataset.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_feature_importance():
    path = "data/processed/feature_importance.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    try:
        model  = joblib.load("models/xgb_f1_predictor.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feats  = joblib.load("models/feature_cols.pkl")
        return model, scaler, feats
    except Exception:
        return None, None, None

preds_df   = load_predictions()
full_df    = load_full_dataset()
imp_df     = load_feature_importance()
model, scaler, feature_cols = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏎️ F1 Race Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏁 Race Predictions", "📊 Model Performance",
         "🔍 Feature Importance", "📈 Driver Form"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if not preds_df.empty:
        n_races   = preds_df.groupby(["year", "round"]).ngroups
        n_drivers = preds_df["driver_number"].nunique()
        st.caption(f"📅 Test season: 2025")
        st.caption(f"🏁 Races evaluated: {n_races}")
        st.caption(f"🧑‍✈️ Drivers: {n_drivers}")

    st.markdown("---")
    st.caption("Built with OpenF1 + XGBoost + Streamlit")


# ── Helper: per-race metrics ──────────────────────────────────────────────────

@st.cache_data
def compute_race_metrics(df):
    rows = []
    for (year, round_), race in df.groupby(["year", "round"]):
        actual_winner    = race[race["position"] == 1]["driver_number"].values
        predicted_winner = race[race["predicted_position"] == 1]["driver_number"].values
        winner_correct   = (
            len(actual_winner) > 0 and len(predicted_winner) > 0
            and actual_winner[0] == predicted_winner[0]
        )

        actual_podium    = set(race[race["position"] <= 3]["driver_number"].tolist())
        predicted_podium = set(race[race["predicted_position"] <= 3]["driver_number"].tolist())
        podium_overlap   = len(actual_podium & predicted_podium) / 3

        from scipy.stats import spearmanr
        merged = race[["position", "predicted_position"]].dropna()
        corr   = spearmanr(merged["position"], merged["predicted_position"])[0] if len(merged) > 2 else 0

        mae = (race["position"] - race["predicted_position"]).abs().mean()

        circuit = race["circuit_short_name"].iloc[0] if "circuit_short_name" in race.columns else f"R{round_}"
        rows.append({
            "year":           year,
            "round":          round_,
            "circuit":        circuit,
            "winner_correct": winner_correct,
            "podium_overlap": podium_overlap,
            "spearman":       corr,
            "mae":            mae,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Race Predictions
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏁 Race Predictions":
    st.title("🏁 Race Predictions")
    st.markdown("Compare predicted vs actual finishing order for any 2025 race.")

    if preds_df.empty:
        st.error("No predictions found. Run `python src/models/train.py` first.")
        st.stop()

    # Race selector
    races = (
        preds_df.groupby(["year", "round", "circuit_short_name"])
        .size().reset_index()
        .sort_values(["year", "round"])
    )
    race_options = {
        f"Round {int(r['round'])} — {r['circuit_short_name']}": (r["year"], r["round"])
        for _, r in races.iterrows()
    }

    selected_label = st.selectbox("Select a race", list(race_options.keys()), index=len(race_options)-1)
    selected_year, selected_round = race_options[selected_label]

    race = preds_df[
        (preds_df["year"] == selected_year) &
        (preds_df["round"] == selected_round)
    ].copy().sort_values("predicted_position")

    # ── Summary metrics for this race ────────────────────────────────────────
    actual_winner    = race[race["position"] == 1]["name_acronym"].values
    predicted_winner = race[race["predicted_position"] == 1]["name_acronym"].values
    winner_correct   = (
        len(actual_winner) > 0 and len(predicted_winner) > 0
        and actual_winner[0] == predicted_winner[0]
    )

    actual_podium    = set(race[race["position"] <= 3]["name_acronym"].tolist())
    predicted_podium = set(race[race["predicted_position"] <= 3]["name_acronym"].tolist())
    podium_hits      = len(actual_podium & predicted_podium)

    from scipy.stats import spearmanr
    corr = spearmanr(race["position"], race["predicted_position"])[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{"✅" if winner_correct else "❌"}</div>
            <div class="metric-label">Winner Predicted</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{podium_hits}/3</div>
            <div class="metric-label">Podium Correct</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{corr:.2f}</div>
            <div class="metric-label">Rank Correlation</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        mae = (race["position"] - race["predicted_position"]).abs().mean()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{mae:.1f}</div>
            <div class="metric-label">Avg Position Error</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side-by-side: predicted order vs actual order ─────────────────────────
    col_pred, col_actual = st.columns(2)

    def position_emoji(pos):
        return {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"P{pos}")

    with col_pred:
        st.markdown('<div class="race-title">🤖 Predicted Order</div>', unsafe_allow_html=True)
        pred_sorted = race.sort_values("predicted_position")
        for _, row in pred_sorted.iterrows():
            p    = int(row["predicted_position"])
            name = row.get("name_acronym", str(row["driver_number"]))
            team = row.get("team_name", "")
            actual_p = int(row["position"])
            diff = actual_p - p
            diff_str = f"(actual P{actual_p}  {'▲' if diff > 0 else '▼' if diff < 0 else '='}{abs(diff) if diff != 0 else ''})"
            color = "#00d26a" if abs(diff) <= 1 else "#ffb700" if abs(diff) <= 3 else "#e10600"
            st.markdown(
                f"`{position_emoji(p)}` **{name}** — {team[:22]}  "
                f"<span style='color:{color}; font-size:0.8rem'>{diff_str}</span>",
                unsafe_allow_html=True
            )

    with col_actual:
        st.markdown('<div class="race-title">🏁 Actual Result</div>', unsafe_allow_html=True)
        actual_sorted = race.sort_values("position")
        for _, row in actual_sorted.iterrows():
            p    = int(row["position"])
            name = row.get("name_acronym", str(row["driver_number"]))
            team = row.get("team_name", "")
            pred_p = int(row["predicted_position"])
            diff   = p - pred_p
            diff_str = f"(predicted P{pred_p})"
            color = "#00d26a" if abs(diff) <= 1 else "#ffb700" if abs(diff) <= 3 else "#e10600"
            st.markdown(
                f"`{position_emoji(p)}` **{name}** — {team[:22]}  "
                f"<span style='color:{color}; font-size:0.8rem'>{diff_str}</span>",
                unsafe_allow_html=True
            )

    # ── Bubble chart: predicted vs actual ────────────────────────────────────
    st.markdown("---")
    st.subheader("Predicted vs Actual Position")

    fig = px.scatter(
        race,
        x         = "predicted_position",
        y         = "position",
        text      = "name_acronym",
        color     = "team_name",
        size      = [10] * len(race),
        template  = "plotly_dark",
        labels    = {"predicted_position": "Predicted Position",
                     "position": "Actual Position"},
    )
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[1, 20], y=[1, 20],
        mode="lines",
        line=dict(color="#e10600", dash="dash", width=1),
        name="Perfect prediction",
        showlegend=True,
    ))
    fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    fig.update_layout(
        height=480,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f0f",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("How well does the model perform across all 2025 races?")

    if preds_df.empty:
        st.error("No predictions found. Run `python src/models/train.py` first.")
        st.stop()

    metrics_df = compute_race_metrics(preds_df)

    # ── Overall metrics ───────────────────────────────────────────────────────
    st.subheader("Overall Metrics")
    col1, col2, col3, col4 = st.columns(4)

    winner_acc   = metrics_df["winner_correct"].mean()
    podium_acc   = metrics_df["podium_overlap"].mean()
    avg_spearman = metrics_df["spearman"].mean()
    avg_mae      = metrics_df["mae"].mean()

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{winner_acc:.1%}</div>
            <div class="metric-label">Winner Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{podium_acc:.1%}</div>
            <div class="metric-label">Podium Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_spearman:.3f}</div>
            <div class="metric-label">Avg Rank Correlation</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_mae:.2f}</div>
            <div class="metric-label">Avg Position Error</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Winner accuracy per race ──────────────────────────────────────────────
    st.subheader("Winner Prediction — Race by Race")
    metrics_df["result"] = metrics_df["winner_correct"].map({True: "✅ Correct", False: "❌ Wrong"})
    metrics_df["color"]  = metrics_df["winner_correct"].map({True: "#00d26a", False: "#e10600"})

    fig = px.bar(
        metrics_df,
        x         = "circuit",
        y         = "podium_overlap",
        color     = "winner_correct",
        color_discrete_map = {True: "#00d26a", False: "#e10600"},
        labels    = {"podium_overlap": "Podium Accuracy", "circuit": "Circuit",
                     "winner_correct": "Winner Correct"},
        template  = "plotly_dark",
        text      = metrics_df["podium_overlap"].apply(lambda x: f"{x:.0%}"),
    )
    fig.update_layout(
        height=380,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f0f",
        xaxis_tickangle=-45,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Spearman rank correlation per race ────────────────────────────────────
    st.subheader("Rank Correlation per Race (higher = better order prediction)")
    fig2 = px.line(
        metrics_df,
        x        = "circuit",
        y        = "spearman",
        markers  = True,
        template = "plotly_dark",
        labels   = {"spearman": "Spearman Correlation", "circuit": "Circuit"},
        color_discrete_sequence = ["#e10600"],
    )
    fig2.add_hline(
        y=avg_spearman,
        line_dash="dash",
        line_color="#ffb700",
        annotation_text=f"Average: {avg_spearman:.3f}",
    )
    fig2.update_layout(
        height=320,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f0f",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw metrics table ─────────────────────────────────────────────────────
    with st.expander("📋 Full metrics table"):
        display = metrics_df[["round", "circuit", "winner_correct",
                               "podium_overlap", "spearman", "mae"]].copy()
        display.columns = ["Round", "Circuit", "Winner ✓", "Podium %", "Spearman", "MAE"]
        display["Podium %"] = display["Podium %"].apply(lambda x: f"{x:.0%}")
        display["Spearman"] = display["Spearman"].round(3)
        display["MAE"]      = display["MAE"].round(2)
        st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("Which features does the model rely on most to make predictions?")

    if imp_df.empty:
        st.error("Feature importance file not found. Run `python src/models/train.py` first.")
        st.stop()

    imp_df = imp_df.sort_values("importance", ascending=True)

    # Clean up feature names for display
    name_map = {
        "driver_champ_rank_before":        "Driver Championship Rank",
        "avg_points_last3":                "Avg Points (Last 3 Races)",
        "constructor_champ_rank_before":   "Constructor Championship Rank",
        "races_completed_this_season":     "Races Completed This Season",
        "rainfall_flag":                   "Rainfall (wet race)",
        "avg_pit_duration":                "Avg Pit Stop Duration",
        "team_name_encoded":               "Team",
        "constructor_champ_pts_before":    "Constructor Championship Points",
        "driver_champ_pts_before":         "Driver Championship Points",
        "avg_finish_last3":                "Avg Finish Position (Last 3 Races)",
        "pit_stop_count":                  "Pit Stop Count",
        "air_temperature":                 "Air Temperature",
        "track_temperature":               "Track Temperature",
        "circuit_short_name_encoded":      "Circuit",
    }
    imp_df["feature_label"] = imp_df["feature"].map(name_map).fillna(imp_df["feature"])

    fig = px.bar(
        imp_df,
        x         = "importance",
        y         = "feature_label",
        orientation = "h",
        template  = "plotly_dark",
        color     = "importance",
        color_continuous_scale = ["#2a2a3e", "#e10600"],
        labels    = {"importance": "Importance Score", "feature_label": ""},
        text      = imp_df["importance"].apply(lambda x: f"{x:.3f}"),
    )
    fig.update_layout(
        height           = 520,
        plot_bgcolor     = "#1a1a2e",
        paper_bgcolor    = "#0f0f0f",
        coloraxis_showscale = False,
        yaxis_title      = "",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 What this tells us")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
**Driver Championship Rank** is the strongest signal —
drivers who are already winning tend to keep winning.
This captures overall car pace + driver skill combined.
        """)
        st.info("""
**Recent Points (last 3 races)** captures momentum —
a driver on a hot streak is more likely to perform well
in the next race too.
        """)
    with col2:
        st.info("""
**Rainfall flag** matters a lot — wet races are unpredictable
and reshuffle the grid, which the model correctly learns to weight.
        """)
        st.info("""
**Team** reflects car performance — some constructors
consistently outperform others at specific circuits.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Driver Form
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Driver Form":
    st.title("📈 Driver Form")
    st.markdown("Track finishing positions and points across the season.")

    if full_df.empty:
        st.error("Dataset not found.")
        st.stop()

    # Filter to 2025
    df_2025 = full_df[full_df["year"] == 2025].copy()

    if df_2025.empty:
        st.warning("No 2025 data available.")
        st.stop()

    # Driver selector
    drivers = sorted(df_2025["name_acronym"].dropna().unique().tolist())
    selected_drivers = st.multiselect(
        "Select drivers to compare",
        drivers,
        default=drivers[:5] if len(drivers) >= 5 else drivers,
    )

    if not selected_drivers:
        st.warning("Please select at least one driver.")
        st.stop()

    df_filtered = df_2025[df_2025["name_acronym"].isin(selected_drivers)]

    # ── Finishing position over the season ────────────────────────────────────
    st.subheader("Finishing Position by Race (lower = better)")
    fig = px.line(
        df_filtered.sort_values("round"),
        x        = "round",
        y        = "position",
        color    = "name_acronym",
        markers  = True,
        template = "plotly_dark",
        labels   = {"round": "Race Round", "position": "Finishing Position",
                    "name_acronym": "Driver"},
    )
    fig.update_yaxes(autorange="reversed")   # P1 at top
    fig.update_layout(
        height=400,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f0f",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Cumulative points ─────────────────────────────────────────────────────
    st.subheader("Cumulative Championship Points")

    POINTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
    df_filtered = df_filtered.copy()
    df_filtered["points"] = df_filtered["position"].map(POINTS_MAP).fillna(0)
    df_filtered = df_filtered.sort_values(["name_acronym", "round"])
    df_filtered["cumulative_points"] = df_filtered.groupby("name_acronym")["points"].cumsum()

    fig2 = px.line(
        df_filtered,
        x        = "round",
        y        = "cumulative_points",
        color    = "name_acronym",
        markers  = True,
        template = "plotly_dark",
        labels   = {"round": "Race Round", "cumulative_points": "Cumulative Points",
                    "name_acronym": "Driver"},
    )
    fig2.update_layout(
        height=380,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f0f",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Season summary table ──────────────────────────────────────────────────
    st.subheader("Season Summary")
    summary = (
        df_filtered.groupby("name_acronym")
        .agg(
            Races       = ("round",    "count"),
            Wins        = ("position", lambda x: (x == 1).sum()),
            Podiums     = ("position", lambda x: (x <= 3).sum()),
            Best_Finish = ("position", "min"),
            Avg_Finish  = ("position", "mean"),
            Total_Pts   = ("points",   "sum"),
        )
        .round(2)
        .sort_values("Total_Pts", ascending=False)
        .reset_index()
        .rename(columns={"name_acronym": "Driver"})
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)
