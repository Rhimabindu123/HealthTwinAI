# HealthTwin AI — Predictive Healthcare Digital Twin

## Overview
HealthTwin AI is a machine learning prototype that forecasts blood glucose levels two hours ahead using synthetic patient profiles and simulated sensor readings. It combines data processing, predictive modelling, and an interactive Streamlit dashboard.

**Disclaimer:** This is an educational prototype using synthetic data. It is not clinically validated and must not be used for medical diagnosis or treatment decisions.

## Problem Statement
Current glucose readings show a person's present state but do not directly indicate future glucose trends. This project explores how machine learning can forecast future readings from historical sensor data and patient information.

## Key Features
- Synthetic patient profile and sensor-data generation.
- Two-hour-ahead glucose forecasting.
- Historical glucose trend visualization.
- Simulation of new sensor readings.
- Comparison with a current-glucose baseline.
- Separate evaluation using patients excluded from model training.

## Technology Stack
- Python
- Scikit-learn and Random Forest Regressor
- Pandas and NumPy
- Streamlit
- Joblib
- CSV and JSON

## Architecture
1. **Data generation:** Creates synthetic patient profiles and hourly glucose and activity readings.
2. **Preprocessing:** Combines patient records and sensor readings and generates historical lag features.
3. **Model training:** Trains a Random Forest regression pipeline to predict glucose two hours ahead.
4. **Evaluation:** Compares the model against a baseline using the current glucose reading.
5. **Dashboard:** Displays patient history, simulated readings, and predictions.

## Preliminary Results

All results below are from synthetic data.

| Evaluation | Model MAE | Baseline MAE | Model RMSE | Baseline RMSE |
|---|---:|---:|---:|---:|
| Temporal holdout | 4.918 | 5.369 | 6.121 | 6.786 |
| Unseen-patient holdout | 4.994 | 5.546 | 6.227 | 6.925 |

The model performed better than the current-glucose baseline in both evaluations. The unseen-patient evaluation held out 20 of the 100 synthetic patients from training.

These results do not establish accuracy on real patients or clinical safety.

## Project Structure

```text
HealthTwinAI/
├── data/
│   ├── patient_records.csv
│   └── sensor_readings.csv
├── models/
│   ├── glucose_model.joblib
│   ├── metrics.json
│   └── unseen_patient_metrics.json
├── app.py
├── generate_data.py
├── train_model.py
├── evaluate_unseen_patients.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Rhimabindu123/HealthTwinAI.git
cd HealthTwinAI
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the synthetic datasets

```bash
python generate_data.py
```

### 5. Train the model

```bash
python train_model.py
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

### 7. Run the unseen-patient evaluation

```bash
python evaluate_unseen_patients.py
```
## Limitations and Future Work

- The dataset is synthetic and may not represent real physiological behaviour.
- The model has not been clinically validated.
- Sensor readings are simulated rather than received from a real wearable device.
- Future work could explore prediction uncertainty, external validation, individualized calibration, and secure wearable integration.

## Disclaimer

HealthTwin AI is an educational prototype, not a medical device. Predictions must not be used to make medical decisions.

## Project Goal

To demonstrate how synthetic health data, machine learning, and an interactive dashboard can be combined in an exploratory digital twin workflow.

## Project Presentation

[View HealthTwin AI Presentation](./HealthTwinAI_Presentation_Rudramoni_Himabindu.pptx)

## Architecture Diagram

[View HealthTwin AI Architecture Diagram](./HealthTwinAI_Architecture_Diagram.pdf)

## College / Incubator Information

- **College:** CMR Engineering College
- **Location:** Hyderabad, Telangana, India
- **Department:** Information Technology
- **Program:** B.Tech
- **Incubator:** Not applicable.

## Team Details

- **Team Name:** HealthTwin AI
- **Team Lead / Sole Contributor:** Rudramoni Himabindu
- **Team Size:** 1
- **Contribution:** Project development, data processing, machine learning, model evaluation, and Streamlit dashboard development.

## Healthcare Use Case

HealthTwin AI demonstrates how machine learning can be used to forecast blood glucose levels two hours ahead using historical glucose readings and synthetic patient data.

The interactive dashboard allows users to:
- Explore synthetic patient profiles and glucose trends.
- View model-generated glucose forecasts.
- Simulate new sensor readings.
- Compare model performance against a baseline.

**Intended use:** Educational research and demonstration of predictive healthcare technology.

**Limitation:** The prototype uses synthetic data, has not been clinically validated, and must not be used for medical diagnosis or treatment decisions.
