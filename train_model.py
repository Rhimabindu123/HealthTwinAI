
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project paths
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load synthetic patient and wearable data
patients = pd.read_csv(DATA_DIR / "patient_records.csv")
sensors = pd.read_csv(DATA_DIR / "sensor_readings.csv")

required_patient_columns = {
    "patient_id", "age", "diabetes_history",
    "baseline_glucose", "activity_level"
}
required_sensor_columns = {
    "patient_id", "hour", "glucose", "steps"
}

if not required_patient_columns.issubset(patients.columns):
    raise ValueError("Patient dataset is missing required columns.")

if not required_sensor_columns.issubset(sensors.columns):
    raise ValueError("Sensor dataset is missing required columns.")

if patients["patient_id"].duplicated().any():
    raise ValueError("Each patient must have exactly one patient record.")

# 2. Fuse historical patient records and sensor readings
df = sensors.merge(
    patients,
    on="patient_id",
    how="inner",
    validate="many_to_one"
)

df = df.sort_values(
    ["patient_id", "hour"]
).reset_index(drop=True)

if df.empty:
    raise ValueError("No matching patient and sensor records found.")

if df.duplicated(["patient_id", "hour"]).any():
    raise ValueError("Duplicate hourly sensor readings were found.")

# Verify that readings are hourly and consecutive
hour_differences = df.groupby("patient_id")["hour"].diff().dropna()

if not hour_differences.eq(1).all():
    raise ValueError(
        "Sensor readings must be consecutive hourly records "
        "for this two-hour forecasting setup."
    )

# 3. Create past-reading features and future target
grouped = df.groupby("patient_id", sort=False)

df["glucose_lag_1"] = grouped["glucose"].shift(1)
df["glucose_lag_2"] = grouped["glucose"].shift(2)
df["steps_lag_1"] = grouped["steps"].shift(1)

# Predict glucose two hours after the current reading
df["target_glucose_2h"] = grouped["glucose"].shift(-2)

df = df.dropna().copy()

# 4. Chronological split
# Training target times end at hour 133.
# Test input times begin at hour 134.
train = df[df["hour"] <= 131].copy()
test = df[df["hour"] >= 134].copy()

if train.empty or test.empty:
    raise ValueError("Not enough data for the chronological split.")

# Explicitly check that training targets do not reach the test period
if train["hour"].max() + 2 >= test["hour"].min():
    raise ValueError("Training targets overlap the test period.")

numeric_features = [
    "age",
    "diabetes_history",
    "baseline_glucose",
    "glucose",
    "glucose_lag_1",
    "glucose_lag_2",
    "steps",
    "steps_lag_1",
]

categorical_features = ["activity_level"]
features = numeric_features + categorical_features
target = "target_glucose_2h"

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# 5. Preprocess and train the model
preprocessor = ColumnTransformer([
    (
        "activity",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    )
])

model.fit(X_train, y_train)

# 6. Evaluate model and persistence baseline
predictions = model.predict(X_test)

# Baseline: assume glucose stays at its current value
baseline_predictions = X_test["glucose"].to_numpy()

def evaluate(actual, predicted):
    return {
        "MAE": round(
            float(mean_absolute_error(actual, predicted)), 3
        ),
        "RMSE": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 3
        ),
        "R2": round(float(r2_score(actual, predicted)), 3)
    }

results = {
    "model": evaluate(y_test, predictions),
    "baseline_current_glucose": evaluate(
        y_test, baseline_predictions
    ),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_last_input_hour": int(train["hour"].max()),
    "testing_first_input_hour": int(test["hour"].min()),
    "forecast_horizon_hours": 2,
    "unique_training_patients": int(
        train["patient_id"].nunique()
    ),
    "unique_testing_patients": int(
        test["patient_id"].nunique()
    ),
    "data_type": "synthetic"
}

# 7. Save model and evaluation results
joblib.dump(model, MODEL_DIR / "glucose_model.joblib")

with open(
    MODEL_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

print("\nModel training completed!")
print("\nEvaluation results:")
print(json.dumps(results, indent=4))
print(f"\nSaved model to: {MODEL_DIR / 'glucose_model.joblib'}")
print(f"Saved metrics to: {MODEL_DIR / 'metrics.json'}")
