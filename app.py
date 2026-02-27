from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("bahrain_hybrid_model.pkl")
feature_columns = joblib.load("bahrain_model_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    lap_number = int(request.form["lap_number"])
    stint = int(request.form["stint"])
    tyre_life = int(request.form["tyre_life"])
    fresh_tyre = int(request.form["fresh_tyre"])
    position = int(request.form["position"])
    compound = request.form["compound"]
    team = request.form["team"]
    driver = request.form["driver"]

    input_data = pd.DataFrame([{
        "Year": year,
        "LapNumber": lap_number,
        "Stint": stint,
        "TyreLife": tyre_life,
        "FreshTyre": fresh_tyre,
        "Position": position,
        "Compound": compound,
        "Team": team,
        "Driver": driver
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["Compound", "Team", "Driver"]
    )

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Lap Time: {round(prediction[0],3)} seconds"
    )

if __name__ == "__main__":
    app.run(debug=True)
