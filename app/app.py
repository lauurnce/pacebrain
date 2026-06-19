"""
Streamlit demo for the finish-time predictor — Day 9.

Same six features and same inference path as the predict.py CLI, wrapped in
a UI so you can drag a slider instead of retyping flags. Nothing is
reimplemented here: the scaling, the forward pass and the time formatting
all come from the package, so the app can never drift from the CLI.

Run from the repo root (the checkpoint path in config.py is relative to it):
    streamlit run app/app.py
"""

import sys
import pathlib

# Same trick as predict.py — the package lives in src/, which isn't on the
# path when Streamlit executes this file directly.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import streamlit as st

from pacebrain.config import FinishPredictorConfig
from pacebrain.inference import load_finish_model, load_scaler, predict_finish_time
from pacebrain.predict import format_hms, format_pace


# The ranges make_sample_data() actually drew from. The model has never seen
# anything outside these, and an MLP extrapolates badly — it will happily
# return a confident-looking number for a 400 km training week. Flag it
# instead of pretending the prediction means something.
# (days_since_long_run uses rng.integers(3, 21), whose upper bound is exclusive.)
TRAINING_RANGES = {
    "weekly_mileage_km": (20.0, 120.0),
    "avg_pace_min_per_km": (4.5, 7.5),
    "long_run_km": (10.0, 35.0),
    "days_since_long_run": (3.0, 20.0),
    "runs_per_week": (3.0, 7.0),
}

LABELS = {
    "weekly_mileage_km": "weekly mileage",
    "avg_pace_min_per_km": "easy-run pace",
    "long_run_km": "longest recent run",
    "days_since_long_run": "days since long run",
    "runs_per_week": "runs per week",
}


@st.cache_resource(show_spinner="Loading model and scaler…")
def load_predictor():
    """
    Load the checkpoint and rebuild the training scaler, once per session.

    Both steps are expensive relative to a widget change: rebuild_scaler
    regenerates the full 1000-row synthetic dataset to recover the exact
    mean/std used in training. Streamlit re-runs this whole script top to
    bottom on every slider move, so without @st.cache_resource that work
    would repeat on every interaction.

    Exceptions are deliberately not caught here — Streamlit doesn't cache a
    failed call, so once the user trains a model the next re-run picks it up
    without needing to clear the cache.
    """
    cfg = FinishPredictorConfig()
    return load_finish_model(cfg), load_scaler(cfg)


def out_of_range(features: dict) -> list:
    """Return human-readable labels for any feature outside the training range."""
    return [
        LABELS[col]
        for col, (low, high) in TRAINING_RANGES.items()
        if not low <= features[col] <= high
    ]


st.set_page_config(page_title="PaceBrain", page_icon="🏃", layout="centered")

st.title("🏃 PaceBrain")
st.write("Predict a race finish time from a recent training block.")

# models/ is gitignored and no checkpoint ships with the repo, so a fresh
# clone lands here. A raw traceback would be a terrible first impression —
# say what to run instead.
try:
    model, scaler = load_predictor()
except FileNotFoundError:
    st.error(
        "No trained model found at `models/finish_predictor.pt`.\n\n"
        "Train one first, from the repo root:\n\n"
        "```bash\n"
        "python src/pacebrain/train_finish.py\n"
        "```\n\n"
        "Then reload this page. (Checkpoints are gitignored, so a fresh "
        "clone never has one.)"
    )
    st.stop()

st.subheader("Your training block")

# Widget bounds are deliberately wider than TRAINING_RANGES so extrapolation
# is possible — and then reported — rather than quietly clamped away.
left, right = st.columns(2)

with left:
    weekly_mileage = st.slider(
        "Weekly mileage (km)", 0.0, 250.0, 60.0, step=1.0,
        help="Average km per week across the training block. Trained on 20–120.",
    )
    long_run = st.slider(
        "Longest recent run (km)", 0.0, 60.0, 28.0, step=0.5,
        help="Longest single run in the block. Trained on 10–35.",
    )
    runs_per_week = st.slider(
        "Runs per week", 1.0, 14.0, 4.0, step=0.5,
        help="Training frequency. Trained on 3–7.",
    )

with right:
    avg_pace = st.slider(
        "Average easy-run pace (min/km)", 3.0, 10.0, 5.5, step=0.05,
        help="Easy pace, not race pace — lower is faster. Trained on 4.5–7.5.",
    )
    days_since_long_run = st.slider(
        "Days since long run", 0.0, 60.0, 7.0, step=1.0,
        help="Recency of that long run. Trained on 3–20.",
    )
    race_distance = st.selectbox(
        "Race distance (km)", [5.0, 10.0, 21.1, 42.2], index=3,
        help="The four distances the model was trained on.",
    )

# Keys must match FEATURE_COLS in data.py — predict_finish_time() reads them
# by name and rebuilds the row in the order the model was trained on.
features = {
    "weekly_mileage_km": weekly_mileage,
    "avg_pace_min_per_km": avg_pace,
    "long_run_km": long_run,
    "days_since_long_run": days_since_long_run,
    "runs_per_week": runs_per_week,
    "race_distance_km": race_distance,
}

minutes = predict_finish_time(model, scaler, features)
race_pace = minutes / race_distance

st.subheader("Prediction")

col_min, col_hms, col_pace = st.columns(3)
col_min.metric("Finish time (min)", f"{minutes:.1f}")
col_hms.metric("Finish time", format_hms(minutes))
col_pace.metric("Implied race pace", f"{format_pace(race_pace)} /km")

extrapolating = out_of_range(features)
if extrapolating:
    st.warning(
        "Outside the training range: "
        + ", ".join(extrapolating)
        + ". The model is extrapolating here, so treat this number as "
        "unreliable rather than merely uncertain."
    )

st.caption(
    "Trained on synthetic data generated by `make_sample_data()`, not real race "
    "results — predictions are illustrative only. See the Results note in the README."
)
