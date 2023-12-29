import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


df = pd.read_csv("perrin-freres-monthly-champagne.csv")
df.isnull().sum()
df.dropna(inplace = True)
df.rename(columns={'Month': 'Date', 'Perrin Freres monthly champagne sales millions ?64-?72': 'Sales'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)

train_date = df.index[-1]
print(train_date)
dti = pd.date_range(train_date, periods=6, freq="M")
print(dti)
pred_start_date = dti[0]
pred_end_date = dti[-1]
print(pred_start_date)
print(pred_end_date)

model_SARIMA = SARIMAX(df['Sales'], order=(2, 1, 4), seasonal_order=(0, 1, 0, 12))
model_SARIMA_fit = model_SARIMA.fit(disp=0, maxiter=200, method='nm')
sarima_prediction = model_SARIMA_fit.predict(start=pred_start_date, end=pred_end_date)
sarima_pickle_path = "sarima_model.pkl"
with open(sarima_pickle_path, 'wb') as sarima_pickle:
    pickle.dump(model_SARIMA_fit, sarima_pickle)

print(sarima_prediction)

model_damped_holt_mul = ExponentialSmoothing(df['Sales'],trend='mul',seasonal='mul',damped_trend=True)
model_holt_mul_fit = model_damped_holt_mul.fit()
expo_smoothing_prediction = model_holt_mul_fit.predict(start = pred_start_date, end = pred_end_date)
expo_smoothing_pickle_path = "expo_smoothing_model.pkl"
with open(expo_smoothing_pickle_path, 'wb') as expo_smoothing_pickle:
    pickle.dump(model_holt_mul_fit, expo_smoothing_pickle)

print(expo_smoothing_prediction)

loaded_sarima_model = pickle.load(open(sarima_pickle_path, 'rb'))
loaded_expo_smoothing_model = pickle.load(open(expo_smoothing_pickle_path, 'rb'))



