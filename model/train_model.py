"""
train_model.py
----------------
Generates a realistic synthetic dataset for food/parcel delivery
and trains a RandomForestRegressor to predict delivery time (minutes).

Run this once to produce `delivery_time_model.pkl`.
The Flask app (app.py) loads that file at request time.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------
# 1. Encoding maps (MUST match the maps used in app.py)
# ---------------------------------------------------------------------
WEATHER_MAP = {'Clear': 0, 'Windy': 1, 'Foggy': 2, 'Rainy': 3, 'Stormy': 4}
TRAFFIC_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
VEHICLE_MAP = {'Bike': 0, 'Scooter': 1, 'Car': 2, 'Van': 3}

WEATHER_PENALTY = {0: 0, 1: 2, 2: 6, 3: 10, 4: 16}       # extra minutes
TRAFFIC_PENALTY = {0: 0, 1: 5, 2: 12, 3: 22}             # extra minutes
VEHICLE_SPEED_KMH = {0: 18, 1: 30, 2: 38, 3: 32}         # avg speed per vehicle

N_SAMPLES = 6000

# ---------------------------------------------------------------------
# 2. Generate synthetic data with a believable underlying formula + noise
# ---------------------------------------------------------------------
distance = np.round(np.random.uniform(0.5, 25, N_SAMPLES), 2)
weather = np.random.choice(list(WEATHER_MAP.values()), N_SAMPLES, p=[0.40, 0.15, 0.15, 0.20, 0.10])
traffic = np.random.choice(list(TRAFFIC_MAP.values()), N_SAMPLES, p=[0.30, 0.35, 0.25, 0.10])
vehicle = np.random.choice(list(VEHICLE_MAP.values()), N_SAMPLES, p=[0.35, 0.30, 0.25, 0.10])
prep_time = np.round(np.random.uniform(2, 30, N_SAMPLES), 1)

travel_time = (distance / np.array([VEHICLE_SPEED_KMH[v] for v in vehicle])) * 60  # minutes
weather_extra = np.array([WEATHER_PENALTY[w] for w in weather])
traffic_extra = np.array([TRAFFIC_PENALTY[t] for t in traffic])

noise = np.random.normal(0, 3.5, N_SAMPLES)

delivery_time = (
    prep_time
    + travel_time
    + weather_extra
    + traffic_extra
    + noise
)
delivery_time = np.clip(delivery_time, 8, None)  # nothing under 8 minutes

df = pd.DataFrame({
    'distance_km': distance,
    'weather': weather,
    'traffic': traffic,
    'vehicle_type': vehicle,
    'prep_time_min': prep_time,
    'delivery_time_min': np.round(delivery_time, 1)
})

# ---------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------
X = df[['distance_km', 'weather', 'traffic', 'vehicle_type', 'prep_time_min']]
y = df['delivery_time_min']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=630,
    max_depth=12,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("=" * 50)
print("Model training complete")
print(f"Mean Absolute Error : {mae:.2f} minutes")
print(f"R^2 Score           : {r2:.3f}")
print("=" * 50)

# ---------------------------------------------------------------------
# 6. Save model
# ---------------------------------------------------------------------
out_path = os.path.join(os.path.dirname(__file__), 'delivery_time_model.pkl')
joblib.dump(model, out_path, compress=3)
print(f"Model saved to: {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")
