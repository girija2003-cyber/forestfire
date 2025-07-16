from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open('regressor.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Redirect root URL to prediction form
@app.route('/')
def index():
    return redirect(url_for('predict_datapoint'))

# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html', result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
