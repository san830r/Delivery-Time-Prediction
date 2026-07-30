"""
app.py
------
Flask application for the Delivery Time Prediction mini-project.

Routes:
    /            -> Home / landing page
    /predict     -> Prediction form (GET)
    /result      -> Handles form submission, runs the ML model, shows result (POST)

The trained scikit-learn model lives at model/delivery_time_model.pkl
and is produced by model/train_model.py.
"""

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load the trained model once, at startup
# ---------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'delivery_time_model.pkl')
model = joblib.load(MODEL_PATH)

# These maps MUST stay identical to the ones used in model/train_model.py
WEATHER_MAP = {'Clear': 0, 'Windy': 1, 'Foggy': 2, 'Rainy': 3, 'Stormy': 4}
TRAFFIC_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
VEHICLE_MAP = {'Bike': 0, 'Scooter': 1, 'Car': 2, 'Van': 3}

WEATHER_OPTIONS = list(WEATHER_MAP.keys())
TRAFFIC_OPTIONS = list(TRAFFIC_MAP.keys())
VEHICLE_OPTIONS = list(VEHICLE_MAP.keys())


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template(
        'predict.html',
        weather_options=WEATHER_OPTIONS,
        traffic_options=TRAFFIC_OPTIONS,
        vehicle_options=VEHICLE_OPTIONS,
        error=None,
        form_data=None
    )


@app.route('/result', methods=['POST'])
def result():
    form_data = request.form

    try:
        distance = float(form_data.get('distance', ''))
        prep_time = float(form_data.get('prep_time', ''))
        weather = form_data.get('weather', 'Clear')
        traffic = form_data.get('traffic', 'Low')
        vehicle = form_data.get('vehicle', 'Bike')

        if distance <= 0 or distance > 200:
            raise ValueError("Distance must be between 0 and 200 km.")
        if prep_time < 0 or prep_time > 120:
            raise ValueError("Preparation time must be between 0 and 120 minutes.")
        if weather not in WEATHER_MAP or traffic not in TRAFFIC_MAP or vehicle not in VEHICLE_MAP:
            raise ValueError("Please choose valid options for weather, traffic and vehicle.")

        features = pd.DataFrame([{
            'distance_km': distance,
            'weather': WEATHER_MAP[weather],
            'traffic': TRAFFIC_MAP[traffic],
            'vehicle_type': VEHICLE_MAP[vehicle],
            'prep_time_min': prep_time
        }])

        predicted_minutes = float(model.predict(features)[0])
        predicted_minutes = max(predicted_minutes, prep_time + 1)  # sanity floor

        hours = int(predicted_minutes // 60)
        minutes = int(round(predicted_minutes % 60))

        return render_template(
            'result.html',
            prediction=round(predicted_minutes, 1),
            hours=hours,
            minutes=minutes,
            distance=distance,
            weather=weather,
            traffic=traffic,
            vehicle=vehicle,
            prep_time=prep_time
        )

    except (ValueError, TypeError) as e:
        message = str(e) if str(e) else "Please fill in every field with a valid value."
        return render_template(
            'predict.html',
            weather_options=WEATHER_OPTIONS,
            traffic_options=TRAFFIC_OPTIONS,
            vehicle_options=VEHICLE_OPTIONS,
            error=message,
            form_data=form_data
        )


if __name__ == '__main__':
    app.run(debug=True)
