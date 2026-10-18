
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")
MODEL_PATH = Path("models/glucose_model.joblib")
PATIENTS_PATH = DATA_DIR / "patient_records.csv"
SENSORS_PATH = DATA_DIR / "sensor_readings.csv"

st.set_page_config(
    page_title="HealthTwin AI",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 HealthTwin AI")
st.caption(
    "Healthcare Digital Twin | Synthetic Data Research Prototype"
)

# Check required files
required_files = [
    PATIENTS_PATH,
    SENSORS_PATH,
    MODEL_PATH,
]

missing = [str(path) for path in required_files if not path.exists()]

if missing:
    st.error(f"Missing required files: {', '.join(missing)}")
    st.stop()

# Load patient data and trained model
patients = pd.read_csv(PATIENTS_PATH)
sensors = pd.read_csv(SENSORS_PATH)
model = joblib.load(MODEL_PATH)

# Keep simulated readings during Streamlit reruns
if "live_readings" not in st.session_state:
    st.session_state.live_readings = {}

if "simulation_step" not in st.session_state:
    st.session_state.simulation_step = {}

# Patient selector
selected_id = st.sidebar.selectbox(
    "Select a synthetic patient",
    patients["patient_id"].tolist(),
)

patient = patients[
    patients["patient_id"] == selected_id
].iloc[0]

history = sensors[
    sensors["patient_id"] == selected_id
].sort_values("hour").copy()

history = history.reset_index(drop=True)

# Add previously simulated readings for this patient
extra_readings = st.session_state.live_readings.get(
    selected_id, []
)

if extra_readings:
    history = pd.concat(
        [history, pd.DataFrame(extra_readings)],
        ignore_index=True,
    )

history = history.sort_values("hour").reset_index(drop=True)

st.subheader(f"Virtual patient: {selected_id}")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Age",
    f"{int(patient['age'])} years",
)

col2.metric(
    "Diabetes history",
    "Yes" if int(patient["diabetes_history"]) == 1 else "No",
)

col3.metric(
    "Activity level",
    str(patient["activity_level"]).title(),
)

# Simulated sensor controls
st.subheader("⌚ Simulated wearable sensor")

st.write(
    "Generate a new synthetic sensor reading to demonstrate "
    "how the virtual patient's data and forecast update."
)

if st.button(
    "➕ Simulate New Sensor Reading",
    type="primary",
):
    latest_row = history.iloc[-1]

    # Small, bounded changes for a demonstration only
    rng = np.random.default_rng()

    new_glucose = float(
        np.clip(
            float(latest_row["glucose"])
            + rng.normal(0, 5),
            65,
            250,
        )
    )

    new_steps = int(
        np.clip(
            float(latest_row["steps"])
            + rng.integers(-100, 301),
            0,
            2000,
        )
    )

    next_hour = int(history["hour"].max()) + 1

    new_reading = {
        "patient_id": selected_id,
        "hour": next_hour,
        "glucose": round(new_glucose, 1),
        "steps": new_steps,
    }

    st.session_state.live_readings.setdefault(
        selected_id, []
    ).append(new_reading)

    st.session_state.simulation_step[selected_id] = (
        st.session_state.simulation_step.get(selected_id, 0) + 1
    )

    st.rerun()

if extra_readings:
    st.info(
        f"Simulated readings added for this patient: "
        f"{len(extra_readings)}"
    )
else:
    st.caption(
        "No new readings simulated in this session yet."
    )

# Historical and simulated readings
st.subheader("📈 Glucose history")

chart_data = history.set_index("hour")[["glucose"]]
st.line_chart(chart_data, y_label="Glucose (mg/dL)")

st.subheader("Latest sensor readings")

latest = history.iloc[-1]

col1, col2 = st.columns(2)

col1.metric(
    "Latest glucose",
    f"{float(latest['glucose']):.1f} mg/dL",
)

col2.metric(
    "Latest recorded steps",
    f"{int(latest['steps'])}",
)

# Build model features from the latest three readings
if len(history) < 3:
    st.error("At least three sensor readings are required.")
    st.stop()

previous = history.iloc[-2]
previous_two = history.iloc[-3]

features = pd.DataFrame([{
    "age": int(patient["age"]),
    "diabetes_history": int(patient["diabetes_history"]),
    "baseline_glucose": float(patient["baseline_glucose"]),
    "glucose": float(latest["glucose"]),
    "glucose_lag_1": float(previous["glucose"]),
    "glucose_lag_2": float(previous_two["glucose"]),
    "steps": int(latest["steps"]),
    "steps_lag_1": int(previous["steps"]),
    "activity_level": str(patient["activity_level"]),
}])

# Predict using the same feature columns as training
prediction = float(model.predict(features)[0])

st.subheader("🔮 Two-hour glucose forecast")

col1, col2 = st.columns(2)

col1.metric(
    "Latest reading",
    f"{float(latest['glucose']):.1f} mg/dL",
)

col2.metric(
    "Predicted in 2 hours",
    f"{prediction:.1f} mg/dL",
    delta=f"{prediction - float(latest['glucose']):+.1f} mg/dL",
)

with st.expander("View model input features"):
    st.dataframe(features, width="stretch")

st.info(
    "This prototype uses synthetic patient and sensor data. "
    "Each simulated reading is randomly generated and is not a "
    "real wearable measurement. Forecasts are not clinically "
    "validated and must not be used for diagnosis or treatment."
)
