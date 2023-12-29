import flask
from flask import Flask, request, render_template, jsonify
import pickle
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


print("app is started")
# Create flask app
flask_app = Flask(__name__, template_folder='.')

flask_app._static_folder = '.'

sarima_model = pickle.load(open("sarima_model.pkl", "rb"))
expo_smoothing_model = pickle.load(open("expo_smoothing_model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index1.html")

@flask_app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == 'POST' or 'GET':
        # Get the uploaded CSV file
        uploaded_file = request.files['file']
        num_months = int(request.form.get('NOM'))

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df.isnull().sum()
                df.dropna(inplace=True)
                df.rename(columns={'Month': 'Date', df.columns[1]: 'Sales'},
                          inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
                train_date = df.index[-1]
                dti = pd.date_range(train_date, periods=num_months, freq="M")
                pred_start_date = dti[0]
                pred_end_date = dti[-1]

                sarima_prediction = sarima_model.predict(start=pred_start_date, end=pred_end_date)
                sarima_prediction = sarima_prediction.to_frame(name='Sarima_Predicted_Sales')
                sarima_prediction.reset_index(inplace=True)
                sarima_prediction.rename(columns={'index': 'Date'}, inplace=True)

                expo_smoothing_prediction = expo_smoothing_model.predict(start=pred_start_date, end=pred_end_date)
                expo_smoothing_prediction = expo_smoothing_prediction.to_frame(name='Expo_Smoothing_Predicted_Sales')
                expo_smoothing_prediction.reset_index(inplace=True)
                expo_smoothing_prediction.rename(columns={'index': 'Date'}, inplace=True)

                combined_predictions = pd.merge(round(sarima_prediction), round(expo_smoothing_prediction), on='Date', how='inner')
                combined_predictions = combined_predictions.reset_index(drop=True)
                print(combined_predictions)
                return render_template('index1.html', tables=[combined_predictions.to_html(classes='data1', index=False)], titles='Predicted Sales')

            except Exception as e:
                return render_template("index1.html", prediction_text=f"Error: {str(e)}")

        else:
            return render_template("index1.html", prediction_text="Please upload a CSV file.")

# Add API endpoint
@flask_app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        # Get the uploaded CSV file
        uploaded_file = request.files['file']
        num_months = int(request.form.get('NOM'))

        if uploaded_file:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(uploaded_file)
            df.isnull().sum()
            df.dropna(inplace=True)
            df.rename(columns={'Month': 'Date', df.columns[1]: 'Sales'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
            train_date = df.index[-1]
            dti = pd.date_range(train_date, periods=num_months, freq="M")
            pred_start_date = dti[0]
            pred_end_date = dti[-1]

            sarima_prediction = sarima_model.predict(start=pred_start_date, end=pred_end_date)
            expo_smoothing_prediction = expo_smoothing_model.predict(start=pred_start_date, end=pred_end_date)

            # Combine predictions
            combined_predictions = pd.DataFrame({
                'Date': sarima_prediction.index,
                'Sarima_Predicted_Sales': sarima_prediction.values,
                'Expo_Smoothing_Predicted_Sales': expo_smoothing_prediction.values
            })

            return jsonify({'predictions': combined_predictions.to_dict(orient='records')})

        else:
            return jsonify({'error': 'Please upload a CSV file.'})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    flask_app.run(debug=True)
    
