# SwiftETA — Delivery Time Prediction

A simple, modern web app that predicts delivery time using a Scikit-learn
machine learning model, served through a Flask backend with a clean,
responsive HTML/CSS/JS frontend.

## Features

- Beautiful animated home page (gradient background + animated delivery route)
- Delivery Time Prediction form (Distance, Weather, Traffic, Vehicle Type, Preparation Time)
- Prediction result page with an animated counter and input breakdown
- Fully responsive (mobile / tablet / desktop)
- Modern cards, icons (Font Awesome), and smooth CSS animations

## Project structure

```
delivery_time_prediction/
├── app.py                     # Flask backend
├── requirements.txt
├── swifteta.html              # Standalone single-file version (no backend needed)
├── model/
│   ├── train_model.py         # Generates data + trains the ML model
│   └── delivery_time_model.pkl  # Trained model (already included)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── result.html
└── static/
    ├── css/style.css
    └── js/script.js
```

## Two ways to use this project

1. **Full Flask + Scikit-learn app** (`app.py` + `templates/` + `static/`) — a real
   trained `RandomForestRegressor` predicts delivery time. Requires Python and
   the dependencies in `requirements.txt`.

2. **Standalone demo** (`swifteta.html`) — the same look, pages and animations,
   but everything (including the prediction formula) runs client-side in
   plain JavaScript. Just double-click the file to open it in a browser —
   no Python, Flask, or install step needed.

## Setup & Run

1. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Retrain the model**

   A trained model is already included at `model/delivery_time_model.pkl`.
   To regenerate it from scratch (e.g. after changing the training logic):

   ```bash
   python model/train_model.py
   ```

4. **Run the Flask app**

   ```bash
   python app.py
   ```

5. Open **http://127.0.0.1:5000** in your browser.

## How the ML model works

`model/train_model.py` builds a synthetic-but-realistic dataset:

- `distance_km` — random distance between 0.5–25 km
- `weather` — Clear / Windy / Foggy / Rainy / Stormy
- `traffic` — Low / Medium / High / Jam
- `vehicle_type` — Bike / Scooter / Car / Van (each with its own average speed)
- `prep_time_min` — kitchen/warehouse preparation time

The target `delivery_time_min` is computed from travel time (distance ÷
vehicle speed) plus preparation time plus weather/traffic penalties, with
random noise added for realism. A **RandomForestRegressor** is trained on
this data (achieves R² ≈ 0.96 on the held-out test set) and saved with
`joblib` as `delivery_time_model.pkl`.

`app.py` loads that model once at startup and uses it to turn form inputs
into a live prediction on the `/result` route.

## Customizing

- Change colors/fonts: edit the CSS variables at the top of `static/css/style.css`.
- Add more vehicle types or weather conditions: update the `*_MAP` dictionaries
  in **both** `app.py` and `model/train_model.py`, then retrain the model.
- Swap in a real historical delivery dataset: replace the synthetic data
  generation section of `train_model.py` with a `pd.read_csv(...)` call using
  the same column names.
