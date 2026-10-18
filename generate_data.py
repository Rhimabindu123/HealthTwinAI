
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# 1. Historical patient profiles
n_patients = 100

patients = pd.DataFrame({
    "patient_id": [f"P{i:03d}" for i in range(1, n_patients + 1)],
    "age": np.random.randint(25, 76, n_patients),
    "diabetes_history": np.random.choice(
        [0, 1], n_patients, p=[0.65, 0.35]
    ),
    "baseline_glucose": np.clip(
        np.random.normal(115, 15, n_patients), 75, 165
    ).round(1),
    "activity_level": np.random.choice(
        ["low", "moderate", "high"], n_patients
    )
})

patients.to_csv(DATA_DIR / "patient_records.csv", index=False)

# 2. Simulate hourly sensor data over seven days
rows = []

for _, patient in patients.iterrows():
    baseline = float(patient["baseline_glucose"])
    glucose = baseline

    for hour in range(168):
        hour_of_day = hour % 24

        # Small daily variation
        daily_pattern = 5 * np.sin(2 * np.pi * hour_of_day / 24)

        # Temporary meal-related increases
        meal_effect = 0
        for meal_hour in [8, 13, 19]:
            distance = min(
                abs(hour_of_day - meal_hour),
                24 - abs(hour_of_day - meal_hour)
            )
            meal_effect += 22 * np.exp(-distance / 2.0)

        # Glucose moves gradually rather than drifting indefinitely
        target = baseline + daily_pattern + meal_effect
        glucose = (
            0.65 * glucose
            + 0.35 * target
            + np.random.normal(0, 4)
        )
        glucose = float(np.clip(glucose, 65, 250))

        steps = int(np.random.randint(0, 1200))

        rows.append({
            "patient_id": patient["patient_id"],
            "hour": hour,
            "glucose": round(glucose, 1),
            "steps": steps
        })

sensors = pd.DataFrame(rows)
sensors.to_csv(DATA_DIR / "sensor_readings.csv", index=False)

print("Improved synthetic datasets generated!")
print(f"Patients: {len(patients)}")
print(f"Sensor readings: {len(sensors)}")
print("\nGlucose summary:")
print(sensors["glucose"].describe().round(2))
print(
    "Percentage at the 250 upper limit:",
    round((sensors["glucose"] >= 249.9).mean() * 100, 2), "%"
)
