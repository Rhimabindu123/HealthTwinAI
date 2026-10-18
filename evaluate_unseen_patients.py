
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load and combine both synthetic data streams
patients = pd.read_csv(DATA_DIR / "patient_records.csv")
sensors = pd.read_csv(DATA_DIR / "sensor_readings.csv")

df = sensors.merge(
    patients,
    on="patient_id",
    how="inner",
    validate="many_to_one"
).sort_values(["patient_id", "hour"]).reset_index(drop=True)

# Build past-only features and a target two hours ahead
grouped = df.groupby("patient_id", sort=False)

df["glucose_lag_1"] = grouped["glucose"].shift(1)
df["glucose_lag_2"] = grouped["glucose"].shift(2)
df["steps_lag_1"] = grouped["steps"].shift(1)
df["target_glucose_2h"] = grouped["glucose"].shift(-2)

df = df.dropna().copy()

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

# Split by patient, not by individual reading
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df[features], df[target], groups=df["patient_id"])
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

# Confirm there is no patient overlap
train_patients = set(train["patient_id"])
test_patients = set(test["patient_id"])

assert train_patients.isdisjoint(test_patients)

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

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

predictions = model.predict(X_test)
baseline = X_test["glucose"].to_numpy()

def evaluate(actual, predicted):
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 3),
        "RMSE": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 3
        ),
        "R2": round(float(r2_score(actual, predicted)), 3)
    }

results = {
    "test_type": "unseen_patients",
    "data_type": "synthetic",
    "forecast_horizon_hours": 2,
    "training_patients": len(train_patients),
    "testing_patients": len(test_patients),
    "training_rows": len(train),
    "testing_rows": len(test),
    "model": evaluate(y_test, predictions),
    "baseline_current_glucose": evaluate(y_test, baseline)
}

output_path = MODEL_DIR / "unseen_patient_metrics.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nUnseen-patient evaluation completed!")
print(json.dumps(results, indent=4))
print(f"\nResults saved to: {output_path}")
