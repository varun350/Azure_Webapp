import os
import logging
import warnings
import contextlib
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from itertools import product
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import pmdarima as pm
from sklearn.model_selection import train_test_split
from scipy import stats
import time
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
# from tbats import TBATS
from scipy.fft import fft
from statsforecast import StatsForecast
from statsforecast.models import ADIDA,IMAPA,CrostonClassic,CrostonOptimized ,TSB
from statsforecast.models import RandomWalkWithDrift,CrostonSBA
# from neuralforecast.models import NBEATS,NHITS,RNN
# from neuralforecast import NeuralForecast
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from sklearn.preprocessing import MinMaxScaler


def train_test_split(data,percent_split):
    unique_ids = data['unique_id'].unique()
    train = pd.DataFrame()
    test= pd.DataFrame()
    for id_value in unique_ids:
        id_data = data[data['unique_id'] == id_value]
        len_id_data=int(len(id_data)*(percent_split/100))
        train_id = id_data.iloc[:len(id_data)-len_id_data]
        test_id = id_data.iloc[len(id_data)-len_id_data:]
        train = pd.concat([train,train_id])
        test = pd.concat([test,test_id])
    train=train.reset_index(drop=True) 
    test=test.reset_index(drop=True)
    return train,test

def hampel_filter(data, window_size, threshold):
    # Convert data to a pandas Series if it's not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    # Calculate the rolling median
    rolling_median = data.rolling(window=window_size, center=True).median()
    # Calculate the Median Absolute Deviation (MAD)
    mad = np.abs(data - rolling_median).rolling(window=window_size, center=True).median()
    # Identify outliers based on the threshold
    outliers = (np.abs(data - rolling_median) > threshold * mad)
    # Replace outliers with the rolling median
    filtered_data = np.where(outliers, rolling_median, data)
    return filtered_data

def check_intermittency(df,col):
    adi_threshold=1.32
    cov_threshold=0.49
    non_zeroes=(df[col]!=0).sum()
    total_periods=len(df)
    ADI=total_periods/non_zeroes
    label=''
    df=df[df[col]!=0]
    COV= (np.std(df[col])/np.mean(df[col]))**2
    if ADI < adi_threshold and COV < cov_threshold:
        label='Smooth'
    if ADI >= adi_threshold and COV < cov_threshold:
        label='Intermittent'
    if ADI < adi_threshold and COV >= cov_threshold:
        label='Erratic'
    if ADI >= adi_threshold and COV >= cov_threshold:
        label='Lumpy'
    return label
    
def intermittency_dict_label(check_df,m_id):
    label=check_intermittency(check_df,m_id)
    return label

# needs to be revised later for pipeline sections
def full_train_test(train,test,intermittency_dict):
    test_check,train_check=pd.DataFrame(),pd.DataFrame()
    for intermittent_k,intermittent_v in intermittency_dict.items():
        train1=train[train['unique_id'].isin(intermittent_v)]
        test1=test[test['unique_id'].isin(intermittent_v)]
        train1['Intermittency_Type']=str(intermittent_k)
        test1['Intermittency_Type']=str(intermittent_k)
        train_check=pd.concat([train_check,train1],axis=0)
        test_check=pd.concat([test_check,test1],axis=0)
    return test_check,train_check


def calculate_prediction_interval(df):
    actuals=df[df['label']=='test']['Actual'].values
    predictions_test_valid=df[(df['label']=='test') | (df['label']=='forecast')]['Forecast'].values
    predictions=df[(df['label']=='test')]['Forecast'].values
    residuals_train = abs(actuals -  predictions)
    std_residuals_train = np.std(residuals_train)
    cf = 0.8
    confidence_interval= cf
    alpha=1-confidence_interval
    multiplier = norm.ppf(1- alpha/2)
    forecast_horizon_factor =np.linspace(1, 2, len(df))
    df['Forecast_Lower'] = np.maximum(predictions_test_valid - (forecast_horizon_factor * multiplier * (std_residuals_train/4)),0)
    df['Forecast_Upper'] = np.maximum(predictions_test_valid + (forecast_horizon_factor * multiplier * (std_residuals_train/4)),0)
    return df

def get_significant_lags(df, ycol, desired_lags=40):
    n = len(df[ycol])
    # Calculate the maximum allowable number of lags (50% of the sample size)
    max_lags = int(n * 0.5)
    # Specify the number of lags as the minimum between the desired value and the maximum allowable value
    nlags = min(desired_lags, max_lags)
    pacf_values = sm.tsa.pacf(df[ycol], nlags=nlags)
    significance_level = 0.05
    confidence_interval = 1.96 / np.sqrt(n)
    significant_lags = [i for i, pacf_value in enumerate(pacf_values) if abs(pacf_value) > confidence_interval]
    significant_lags_v2 = [i for i in significant_lags if i not in [0, 1]]
    # If no significant lags are found, provide default values (e.g., [4, 7])
    if len(significant_lags_v2) == 0:
        significant_lags_v2 = [4, 7]
    return significant_lags_v2

def SES(df_try,forecast_length,forecast_df):
    try:
        alpha=0.05
        model_ses=SimpleExpSmoothing(df_try)
        logging.info("Fitting Simple Exponential Smoothing model...")
        fit_ses=model_ses.fit()      
        logging.info("Model fit successfully.")
        
        logging.info(f"Forecasting {forecast_length} steps ahead...")
        forecast_df['SES']=fit_ses.forecast(steps=(forecast_length)).values
        logging.info(f"Forecasting completed for {forecast_length} steps.")
        forecast_error_std = np.std(fit_ses.resid)
        logging.info("SES function executed successfully.")
        logging.info("")  # Add an empty line
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in SES: {e}")
        return pd.DataFrame()


def DES(df_try,forecast_length,forecast_df):
    try:
        model_des=ExponentialSmoothing(df_try,trend='add')

        logging.info("Fitting DES model...")
        fit_des=model_des.fit()
        logging.info("Model fit successfully.")

        logging.info(f"Forecasting {forecast_length} steps ahead...")
        forecast_df['DES']=fit_des.forecast(steps=(forecast_length)).values
        logging.info(f"Forecasting completed for {forecast_length} steps.")

        forecast_error_std = np.std(fit_des.resid)
        logging.info("DES function executed successfully.")
        logging.info("")  # Add an empty line
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in DES: {e}")
        return pd.DataFrame()

def TES(df_try,forecast_length,forecast_df,seasonal_period,initial_method,col):
    try:
        logging.info("Fitting TES model...")
        best_model = None
        best_seasonal_period = None
        best_forecast_error_std = float('inf')
        for seasonal_p in seasonal_period:
            model_tes = ExponentialSmoothing(df_try, trend='add', seasonal='add', seasonal_periods=seasonal_p, initialization_method=initial_method)
            fit_tes = model_tes.fit()
            forecast_values = fit_tes.forecast(steps=forecast_length).values
            forecast_error_std = np.std(fit_tes.resid)
        
            if forecast_error_std < best_forecast_error_std:
                best_model = fit_tes
                best_seasonal_period = seasonal_p
                best_forecast_error_std = forecast_error_std
                if len(col)==0:
                    forecast_df['TES']=forecast_values
                    # forecast_df['best_seasonal_period'] = best_seasonal_period
                else :
                    forecast_df['TES_'+col]=forecast_values
                    # forecast_df['best_seasonal_period'] = best_seasonal_period
                
                # Update forecast_df with the best model's forecast and prediction intervals
        logging.info("TES function executed successfully.")
        logging.info("") 
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in TES: {e}")
        return pd.DataFrame()


def CROSTON_f(df_try,forecast_length,forecast_df):
    try:
        df_try_2=df_try.rename(columns={'Values':'y','Date':'ds'})
        temp_df=None
        model=StatsForecast(models=[ADIDA(),CrostonClassic(),CrostonOptimized(),IMAPA(),TSB(alpha_d=0.2,alpha_p=0.2),RandomWalkWithDrift(),CrostonSBA()],freq='M',n_jobs=1)
        model.fit(df_try_2)
        temp_df=model.predict(h=forecast_length)   
        forecast_df['ADIDA']=temp_df['ADIDA'].values
        forecast_df['CrostonClassic']=temp_df['CrostonClassic'].values
        forecast_df['IMAPA']=temp_df['IMAPA'].values
        forecast_df['TSB']=temp_df['TSB'].values
        forecast_df['CrostonOptimised']=temp_df['CrostonOptimized'].values
        forecast_df['RWD']=temp_df['RWD'].values
        forecast_df['CrostonSBA']=temp_df['CrostonSBA'].values
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in CROSTON_f: {e}")
        return pd.DataFrame()

    
def NBEATS_model(df_try,forecast_length,forecast_df,m_steps):
    try:
        df_try=df_try.rename(columns={'Date':'ds','Values':'y'})
        models=[NBEATS(input_size=2*forecast_length,h=forecast_length,max_steps=m_steps,start_padding_enabled=True),NHITS(input_size=2*forecast_length,h=forecast_length,max_steps=m_steps,start_padding_enabled=True)]
        nf=NeuralForecast(models=models,freq='M')
        nf.fit(df_try)
        temp_df=nf.predict()
        forecast_df['NBEATS']=temp_df['NBEATS'].values
        forecast_df['NHITS']=temp_df['NHITS'].values
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in NBEATS: {e}")
        return pd.DataFrame()

def ARMA_check(df_try, forecast_length, forecast_df, col):
    try:
        best_aic = float('inf')
        best_order = None

        # Define the range of p, d, q values
        p_values = range(0, 5)  # AR parameter
        q_values = range(0, 5)  # MA parameter

        # Generate all possible combinations of p, d, q
        parameters = product(p_values, q_values)

        for param in parameters:
            try:
                model = sm.tsa.ARIMA(df_try, order=(param[0], 0, param[1]))
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = param
            except:
                continue

        # Use the best parameters found
        p, q = best_order
        print(p,q)
        model = sm.tsa.ARIMA(df_try, order=(p,0,q))
        model_fit = model.fit()
        arma_forecast_results = model_fit.predict(start=0, end=forecast_length-1)
        
        if len(col) == 0:
            forecast_df['ARMA'] = arma_forecast_results.values
        else:
            forecast_df['ARMA_' + col] = arma_forecast_results.values
        
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in ARMA: {e}")
        return pd.DataFrame()

def ARIMA_check(df_try,forecast_length,forecast_df,col):
    try:
        with contextlib.redirect_stdout(None):
            autoarima_model=pm.auto_arima(df_try,seasonal=False,stepwise=True,trace=True,suppress_warnings=True,error_action="ignore")
            p,d,q=autoarima_model.order
            model_arima=pm.ARIMA(order=(p,d,q),seasonal=False)

            logging.info("Fitting ARIMA model...")
            model_arima.fit(df_try)
            logging.info("Model fit successfully.")

        logging.info(f"Forecasting {forecast_length} steps ahead...")
        arima_forecast_results,conf_int_arima=model_arima.predict(n_periods=(forecast_length),return_conf_int=True)
        logging.info(f"Forecasting completed for {forecast_length} steps.")
        if len(col)==0:
            forecast_df['ARIMA']=arima_forecast_results.values
        else :
            forecast_df['ARIMA_'+col]=arima_forecast_results.values
            
        logging.info("ARIMA function executed successfully.")
        logging.info("")  # Add an empty line
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in ARIMA: {e}")
        return pd.DataFrame()

def SARIMA_check(df_try,forecast_length,forecast_df,col):
    try:
        logging.info("Fitting SARIMA model...")
        model_fit = pm.auto_arima(df_try, start_p=1, start_q=1,
                             test='adf',
                             max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=True,
                             d=None, D=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
        logging.info("Model fit successfully.")

        logging.info(f"Forecasting {forecast_length} steps ahead...")
        sarima_forecast_results=model_fit.predict(n_periods=(forecast_length))
        logging.info(f"Forecasting completed for {forecast_length} steps.")
         
        if len(col)==0:
            forecast_df['SARIMA']=sarima_forecast_results.values
        else :
            forecast_df['SARIMA_'+col]=sarima_forecast_results.values
        # forecast_df['SARIMA']=sarima_forecast_results.values
        logging.info("SARIMA function executed successfully.")
        logging.info("")  # Add an empty line
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in SARIMA: {e}")
        return pd.DataFrame()

def TBATS_s(df_try,forecast_length,forecast_df):
    try:
        tbats_model=TBATS(seasonal_periods=[12])
        tbats_fit=tbats_model.fit(df_try)
        forecast_df['TBATS']=tbats_fit.forecast(steps=forecast_length)
        return forecast_df
    except Exception  as e:
        print(f"An error occurred in TBATS_s: {e}")
        return pd.DataFrame()


def SKFORECAST_XGB(df_try,test_df_try,forecast_length,forecast_df,lags_grid):
    try:
        params_dict=dict()
        forecast_horizon=7
        initial_train_size=int(len(df_try)*0.7)
        len_test=len(test_df_try)
        with contextlib.redirect_stdout(None):
            sk_model=ForecasterAutoreg(regressor=xgb.XGBRegressor(verbose=False),lags=int(len(df_try)*0.5))
            df_try=df_try.reset_index()
            param_grid={'regressor_n_estimators':[50,75,150,200],
                'regressor_max_depth':[1,2],
                'regressor_learning_rate':[0.1]}
            best_forecaster=grid_search_forecaster(forecaster=sk_model,   
                y                  = df_try['Values'],
                param_grid         = param_grid,
                lags_grid          = lags_grid,
                steps              = forecast_length,
                refit              = False,
                metric             = 'mean_absolute_percentage_error',
                initial_train_size = int(len(df_try)*0.7),
                fixed_train_size   = False,
                return_best        = True,
                n_jobs             = 'auto',
                verbose            = False
                                            
                )
        logging.info("Fitting SKFORECAST_XGB model...")
        sk_model.fit(df_try['Values'])
        logging.info("Model fit successfully.")

        logging.info(f"Forecasting {forecast_length} steps ahead...")
        predictions=sk_model.predict(steps=(forecast_length))
        logging.info(f"Forecasting completed for {forecast_length} steps.")

        predictions=pd.DataFrame(predictions)
        test_df_try = test_df_try.reset_index(drop=True)
        predictions = predictions.reset_index(drop=True)
        residuals=test_df_try['Values']-predictions['pred'][:len_test]
        # forecast_error_std=np.std(residuals)
        # lower_bound,upper_bound=calculate_prediction_interval(predictions['pred'][:],forecast_error_std)
        
        forecast_df['SKFORECAST_XGB']=predictions['pred']
        # forecast_df['SKFORECAST_XGB_Lower']=lower_bound
        # forecast_df['SKFORECAST_XGB_Upper']=upper_bound
        
        best_params=best_forecaster.iloc[0]['params']
        best_lags=best_forecaster.iloc[0]['lags']
        params_dict.update({'params':best_params,'lags':best_lags})
        
        logging.info("SKFORECAST_XGB function executed successfully.")
        logging.info("")  # Add an empty line
        return forecast_df,params_dict
    except Exception  as e:
        print(f"An error occurred in SKFORECAST_XGB: {e}")
        return pd.DataFrame(),{}
    

def SKFORECAST_LGB(df_try,test_df_try,forecast_length,forecast_df,lags_grid):
    try:
        params_dict=dict()
        # forecast_horizon=7
        initial_train_size=int(len(df_try)*0.7)
        len_test=len(test_df_try)
        with contextlib.redirect_stdout(None):
            sk_model_lgb=ForecasterAutoreg(regressor=lgb.LGBMRegressor(verbose=-1),lags=int(len(df_try)*0.5))
            df_try=df_try.reset_index()
            param_grid={'regressor_n_estimators':[3000],
                # 'regressor_max_depth':[4],
                'boosting_type': ['gbdt'],
                'regressor_learning_rate':[0.015],
                'metric': ['mape'],
                'objective':  ['poisson'],
                'seed':  [200],
                'force_row_wise' :  [True],
                'lambda':  [0.1],
                'num_leaves':  [63],
                'sub_row' :  [0.7],
                'bagging_freq' :  [1],
                'colsample_bytree':  [0.7]}
            best_forecaster=grid_search_forecaster(forecaster=sk_model_lgb,
        
                y                  = df_try['Values'],
                param_grid         = param_grid,
                lags_grid          = lags_grid,
                steps              = forecast_length,
                refit              = False,
                metric             = 'mean_absolute_percentage_error',
                initial_train_size = int(len(df_try)*0.8),
                fixed_train_size   = False,
                return_best        = True,
                n_jobs             = 'auto',
                verbose            = False
                                            
                )
        logging.info("Fitting SKFORECAST_LGB model...")
        sk_model_lgb.fit(df_try['Values'])
        logging.info("Model fit successfully.")

        logging.info(f"Forecasting {forecast_length} steps ahead...")
        predictions=sk_model_lgb.predict(steps=(forecast_length))
        logging.info(f"Forecasting completed for {forecast_length} steps.")

        predictions=pd.DataFrame(predictions)
        test_df_try = test_df_try.reset_index(drop=True)
        predictions = predictions.reset_index(drop=True)
        residuals=test_df_try['Values']-predictions['pred'][:len_test]
        forecast_error_std=np.std(residuals)
        # lower_bound,upper_bound=calculate_prediction_interval(predictions['pred'][:],forecast_error_std)
        forecast_df['SKFORECAST_LGB']=predictions['pred']
        # forecast_df['SKFORECAST_LGB_Lower']=lower_bound
        # forecast_df['SKFORECAST_LGB_Upper']=upper_bound
        best_params=best_forecaster.iloc[0]['params']
        best_lags=best_forecaster.iloc[0]['lags']
        params_dict.update({'params':best_params,'lags':best_lags})
        
        logging.info("SKFORECAST_LGB function executed successfully.")
        logging.info("")  # Add an empty line
        return forecast_df,params_dict
    except Exception  as e:
        print(f"An error occurred in SKFORECAST_LGB: {e}")
        return pd.DataFrame(),{}


def SKFORECAST_Catboost(df_try,test_df_try,forecast_length,forecast_df,lags_grid):
    try:
        params_dict=dict()
        forecast_horizon=7
        initial_train_size=int(len(df_try)*0.7)
        len_test=len(test_df_try)
        with contextlib.redirect_stdout(None):
            sk_model_cb=ForecasterAutoreg(regressor=cb.CatBoostRegressor(silent=True),lags=int(len(df_try)*0.5))
            df_try=df_try.reset_index()
            param_grid={'n_estimators':[100,400,500],
                'depth':[4,5],
                'learning_rate':[0.1]}
            best_forecaster=grid_search_forecaster(forecaster=sk_model_cb, 
                y                  = df_try['Values'],
                param_grid         = param_grid,
                lags_grid          = lags_grid,
                steps              = forecast_length,
                refit              = False,
                metric             = 'mean_absolute_percentage_error',
                initial_train_size = int(len(df_try)*0.7),
                fixed_train_size   = False,
                return_best        = True,
                n_jobs             = 'auto',
                verbose            = False
                                            
                )
            
        logging.info("Fitting SKFORECAST_Catboost model...")
        sk_model_cb.fit(df_try['Values'])
        logging.info("Model fit successfully.")

        logging.info(f"Forecasting {forecast_length} steps ahead...")
        predictions=sk_model_cb.predict(steps=(forecast_length))
        logging.info(f"Forecasting completed for {forecast_length} steps.")

        predictions=pd.DataFrame(predictions)
        test_df_try = test_df_try.reset_index(drop=True)
        predictions = predictions.reset_index(drop=True)
        residuals=test_df_try['Values']-predictions['pred'][:len_test]
        # forecast_error_std=np.std(residuals)
        # lower_bound,upper_bound=calculate_prediction_interval(predictions['pred'][:],forecast_error_std)
        forecast_df['SKFORECAST_CATBOOST']=predictions.values
        # forecast_df['SKFORECAST_CATBOOST_Lower']=lower_bound
        # forecast_df['SKFORECAST_CATBOOST_Upper']=upper_bound
        best_params=best_forecaster.iloc[0]['params']
        best_lags=best_forecaster.iloc[0]['lags']
        params_dict.update({'params':best_params,'lags':best_lags})

        logging.info("SKFORECAST_Catboost function executed successfully.")
        logging.info("")  # Add an empty line
        return forecast_df,params_dict
    except Exception  as e:
        print(f"An error occurred in SKFORECAST_Catboost: {e}")
        return pd.DataFrame(),{}

def SKFORECAST_HistGradboost(df_try,test_df_try,forecast_length,forecast_df,lags_grid):
    try:
        params_dict=dict()
        forecast_horizon=7
        initial_train_size=int(len(df_try)*0.7)
        len_test=len(test_df_try)
        with contextlib.redirect_stdout(None):
            sk_model_hgb=ForecasterAutoreg(regressor=HistGradientBoostingRegressor(verbose=0),lags=int(len(df_try)*0.5))
            df_try=df_try.reset_index()
            #sk_model.fit(df_try)
            param_grid={'max_iter':[100,200,400],
                'max_depth':[3,4],
                'learning_rate':[0.1]}
            best_forecaster=grid_search_forecaster(forecaster=sk_model_hgb,   
                y                  = df_try['Values'],
                param_grid         = param_grid,
                lags_grid          = lags_grid,
                steps              = forecast_length,
                refit              = False,
                metric             = 'mean_absolute_percentage_error',
                initial_train_size = int(len(df_try)*0.7),
                fixed_train_size   = False,
                return_best        = True,
                n_jobs             = 'auto',
                verbose            = False
                                            
                )
            
        logging.info("Fitting SKFORECAST_HistGradboost model...")
        sk_model_hgb.fit(df_try['Values'])
        logging.info("Model fit successfully.")
        
        logging.info(f"Forecasting {forecast_length} steps ahead...")
        predictions=sk_model_hgb.predict(steps=(forecast_length))
        logging.info(f"Forecasting completed for {forecast_length} steps.")

        
        predictions=pd.DataFrame(predictions)
        test_df_try = test_df_try.reset_index(drop=True)
        predictions = predictions.reset_index(drop=True)
        residuals=test_df_try['Values']-predictions['pred'][:len_test]
        # forecast_error_std=np.std(residuals)
        # lower_bound,upper_bound=calculate_prediction_interval(predictions['pred'][:],forecast_error_std)
        forecast_df['SKFORECAST_HISTGRADBOOST']=predictions['pred']
        # forecast_df['SKFORECAST_HISTGRADBOOST_Lower']=lower_bound
        # forecast_df['SKFORECAST_HISTGRADBOOST_Upper']=upper_bound
        best_params=best_forecaster.iloc[0]['params']
        best_lags=best_forecaster.iloc[0]['lags']
        params_dict.update({'params':best_params,'lags':best_lags})

        logging.info("SKFORECAST_HistGradboost function executed successfully.")
        logging.info("")  # Add an empty line
        return forecast_df,params_dict
    except Exception  as e:
        print(f"An error occurred in SKFORECAST_HistGradboost: {e}")
        return pd.DataFrame(),{}
    

def Hybrid_DES_SKFORECASTXGB(df_try,test_df_try,forecast_length,forecast_df,master_forecast_df):
    forecast_df2=pd.DataFrame()
    model_1=['TES','ARIMA','SARIMA','ARMA']
    model_2=['SKFORECAST_XGB','SKFORECAST_LGB','SKFORECAST_CATBOOST','SKFORECAST_HISTGRADBOOST']
    
    model1_lst=[col for col in master_forecast_df.columns if col in model_1]
    model2_lst=[col for col in forecast_df.columns if col in model_2]
    print('model1_lst',model1_lst)
    print('model2_lst',model2_lst)
    weight_des=0.5
    weight_xgb=0.5
    len_test=len(test_df_try)
    for m1 in model1_lst:
        for m2 in model2_lst:
            try:
                forecast_df2['HYBRID_'+m1+'_'+m2]=(weight_des*master_forecast_df[m1].values)+(weight_xgb*forecast_df[m2].values)
                # predictions=forecast_df1['HYBRID_'+m1+'_'+m2]
                # predictions=pd.DataFrame(predictions)
                # test_df_try = test_df_try.reset_index(drop=True)
                # predictions = predictions.reset_index(drop=True)
                # residuals=test_df_try['Values']-predictions['HYBRID_'+m1+'_'+m2][:len_test]
                # forecast_error_std=np.std(residuals)
                # lower_bound,upper_bound=calculate_prediction_interval(predictions['HYBRID_'+m1+'_'+m2][:],forecast_error_std)
                # forecast_df['HYBRID_'+m1+'_'+m2+'_Lower']=lower_bound
                # forecast_df['HYBRID_'+m1+'_'+m2+'_Upper']=upper_bound
            except Exception  as e:
                print(f"An error occurred in Hybrid_DES_SKFORECASTXGB while forcasting 'HYBRID_'+m1+'_'+m2 : {e}")
                
    logging.info("Hybrid_DES_SKFORECASTXGB function executed successfully.")
    logging.info("")  # Add an empty line
    return forecast_df2

def models_pipeline1(df_try,forecast_length,seasonal_period,initial_method):
    # forecast_SES = pd.DataFrame()
    # forecast_DES = pd.DataFrame()
    # forecast_TES = pd.DataFrame()
    # forecast_ARIMA = pd.DataFrame()
    # forecast_SARIMA = pd.DataFrame()
    # forecast_ARMA = pd.DataFrame()
    # st_ses=time.time()
    # forecast_SES=SES(df_try,forecast_length,pd.DataFrame())
    # end_ses=time.time()
    # print("SES time",end_ses-st_ses)
    # st_des=time.time()
    # forecast_DES=DES(df_try,forecast_length,pd.DataFrame())
    # end_des=time.time()
    # print("DES time",end_des-st_des)
    
    st_tes=time.time()
    forecast_TES=TES(df_try,forecast_length,pd.DataFrame(),seasonal_period,initial_method,'')
    end_tes=time.time()
    print("TES time",end_tes-st_tes)
    st_arima=time.time()
    forecast_ARIMA=ARIMA_check(df_try,forecast_length,pd.DataFrame(),'')
    end_arima=time.time()
    print("ARIMA time",end_arima-st_arima)
    st_sarima=time.time()
    forecast_SARIMA=SARIMA_check(df_try,forecast_length,pd.DataFrame(),'')
    end_sarima=time.time()
    
    print("ARMA time",end_arima-st_arima)
    st_sarima=time.time()
    forecast_ARMA=ARMA_check(df_try,forecast_length,pd.DataFrame(),'')
    end_sarima=time.time()
    
    print("SARIMA time",end_sarima-st_sarima)
    resultant_df=pd.DataFrame()
    resultant_df=pd.concat([forecast_TES,forecast_ARIMA,forecast_SARIMA,forecast_ARMA],axis=1)
    return resultant_df

def pipeline1_forecast (train_check,test_check,fin_id, no_months_forecast,seasonal_period,initial_method):
    df_try=train_check[train_check.unique_id==fin_id].copy()
    df_try['Values']=df_try['Values'].astype(float).round(2)
    test_df_try=test_check[test_check.unique_id==fin_id]
    forecast_length=len(test_df_try)+no_months_forecast
    forecast_df=models_pipeline1(df_try['Values'],forecast_length,seasonal_period,initial_method)
    forecast_df['Intermittency_check']=test_df_try['Intermittency_Type'].values[-1]
    forecast_df['unique_id']=fin_id
    forecast_df['Actual'] = np.nan  # Initialize the target column with NaN
    n=test_df_try.shape[0]
    
    forecast_df.loc[:n, 'Actual'] = test_df_try['Values'].reset_index(drop=True)
    
    df=test_df_try[['Date']]
    last_date = test_df_try['Date'].iloc[-1]
    extended_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=no_months_forecast, freq='MS')
    extended_df = pd.DataFrame({'Date': extended_dates})
    result_df = pd.concat([df, extended_df], ignore_index=True)
    forecast_df['Date']=result_df['Date']
    print("forecast_df",forecast_df)
    return forecast_df


def models_pipeline2(df_try,test_df_try,forecast_length,masterfrcst1_temp,lags_grid):
    # forecast_df_SKFORECAST_XGB=pd.DataFrame()
    # forecast_df_SKFORECAST_LGB=pd.DataFrame()
    # forecast_df_SKFORECAST_Catboost=pd.DataFrame()
    # forecast_df_SKFORECAST_HistGradboost=pd.DataFrame()
    # forecast_df_Hybrid_DES_SKFORECASTXGB=pd.DataFrame()
    st=time.time()
    forecast_df_SKFORECAST_XGB,xgb_bestparams=SKFORECAST_XGB(df_try,test_df_try,forecast_length,pd.DataFrame(),lags_grid)
    end=time.time()
    print("SKFORECAST_XGB time",end-st)

    st=time.time()
    forecast_df_SKFORECAST_LGB,lgb_bestparams =SKFORECAST_LGB(df_try,test_df_try,forecast_length,pd.DataFrame(),lags_grid)
    end=time.time()
    print("SKFORECAST_LGB time",end-st)
    
    st=time.time()
    forecast_df_SKFORECAST_Catboost,catboost_bestparams=SKFORECAST_Catboost(df_try,test_df_try,forecast_length,pd.DataFrame(),lags_grid)
    end=time.time()
    print("SKFORECAST_Catboost time",end-st)
    
    st=time.time()
    forecast_df_SKFORECAST_HistGradboost,histgradboost_bestparams=SKFORECAST_HistGradboost(df_try,test_df_try,forecast_length,pd.DataFrame(),lags_grid)
    end=time.time()
    print("SKFORECAST_HistGradboost time",end-st)

    forecast_df1=pd.concat([forecast_df_SKFORECAST_XGB,forecast_df_SKFORECAST_LGB,forecast_df_SKFORECAST_Catboost,forecast_df_SKFORECAST_HistGradboost],axis=1)
    print('forecast_df1',forecast_df1)
    
    st=time.time()
    forecast_df_Hybrid_DES_SKFORECASTXGB=Hybrid_DES_SKFORECASTXGB(df_try,test_df_try,forecast_length,forecast_df1,masterfrcst1_temp)
    end=time.time()
    print("Hybrid_DES_SKFORECASTXGB time",end-st)
    
    forecast_df=pd.concat([forecast_df1,forecast_df_Hybrid_DES_SKFORECASTXGB],axis=1)
    
    print('models_pipeline2',forecast_df)
    # return forecast_df,xgb_bestparams,lgb_bestparams,catboost_bestparams,histgradboost_bestparams
    return forecast_df

def pipeline2_forecast(train_check,test_check,master_forecast_df,fin_id,params_struct,no_months_forecast):
    df_try=train_check[train_check.unique_id==fin_id]
    reg_df_try=df_try.copy()
    df_try['Values']=df_try['Values'].astype(float).round(2)
    test_df_try=test_check[test_check.unique_id==fin_id]
    # forecast_df=pd.DataFrame()
    forecast_length=len(test_df_try)+no_months_forecast
    reg_df_try=reg_df_try[['Date','Values']]
    masterfrcst1_temp=master_forecast_df[master_forecast_df.unique_id==fin_id].copy()
    lags_grid=get_significant_lags(df_try,'Values')
    # forecast_df,xgb_bestparams,lgb_bestparams,catboost_bestparams,histgradboost_bestparams=models_pipeline2(df_try['Values'],test_df_try,forecast_length,forecast_df,masterfrcst1_temp,lags_grid)
    forecast_df=pd.DataFrame()
    forecast_df=models_pipeline2(df_try['Values'],test_df_try,forecast_length,masterfrcst1_temp,lags_grid)
   
    
    # params_struct=params_struct._append({'unique_id':fin_id,'XGB':xgb_bestparams,'LGB':lgb_bestparams,'CATBOOST':catboost_bestparams,'HISTGRADBOOST':histgradboost_bestparams},ignore_index=True)
    forecast_df['Intermittency_check']=test_df_try['Intermittency_Type'].values[-1]
    forecast_df['unique_id']=fin_id
    forecast_df['Actual'] = np.nan  # Initialize the target column with NaN
    n=test_df_try.shape[0]
    
    forecast_df.loc[:n, 'Actual'] = test_df_try['Values'].reset_index(drop=True)
    df=test_df_try[['Date']]
    last_date = test_df_try['Date'].iloc[-1]
    extended_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=no_months_forecast, freq='MS')
    extended_df = pd.DataFrame({'Date': extended_dates})
    result_df = pd.concat([df, extended_df], ignore_index=True)
    forecast_df['Date']=result_df['Date']
    return forecast_df

def removal_zeroes(train_df):
    first_index=0
    for i,ele in enumerate(train_df):
        if ele>0:
            first_index=i
            break   
    return first_index
    
def decomposition(train_check_after_zeroes_r,train_sd):
    for fin_id in train_check_after_zeroes_r.unique_id.unique():
        try:
            df_try=train_check_after_zeroes_r[train_check_after_zeroes_r.unique_id==fin_id].copy()
            df_try.set_index(['Date'],inplace=True)
            result=STL(df_try['Values'],seasonal=13,period=12)
            decomposition=result.fit()
            trend=decomposition.trend
            seasonal=decomposition.seasonal
            resid=decomposition.resid
            df_try['Trend']=trend.fillna(0)
            df_try['Seasonal']=seasonal
            df_try['Residual']=resid.fillna(0)
            df_try['unique_id']=fin_id
            resis=resid.fillna(0)
            df_try.reset_index(inplace=True)
            train_sd=pd.concat([train_sd,df_try],axis=0)
        except Exception  as e:
            print(f"An error occurred in decomposition for {fin_id}: {e}")
    return train_sd

def decompose(df_try,forecast_length,seasonal_period,initial_method):
    # st=time.time()
    # forecast_df=SES(df_try['Trend'],forecast_length,forecast_df)
    # end=time.time()
    # print("decompose_ses",end-st)

    st=time.time()
    # forecast_df_tes_trend=pd.DataFrame()
    # forecast_df_tes_Seasonal=pd.DataFrame()
    # forecast_df_tes_Residual=pd.DataFrame() 
    forecast_df=pd.DataFrame()
    
    forecast_df_tes_trend=TES(df_try['Trend'],forecast_length,pd.DataFrame(),seasonal_period,initial_method,'Trend')
    forecast_df_tes_Seasonal=TES(df_try['Seasonal'],forecast_length,pd.DataFrame(),seasonal_period,initial_method,'Seasonal')
    forecast_df_tes_Residual=TES(df_try['Residual'],forecast_length,pd.DataFrame(),seasonal_period,initial_method,'Residual')
    end=time.time()
    
    print("decompose_tes",end-st)
    len_df_tsr=len(forecast_df_tes_trend) + len(forecast_df_tes_Seasonal) + len(forecast_df_tes_Residual)
    if len_df_tsr>0:
        forecast_df = pd.concat([forecast_df_tes_trend,forecast_df_tes_Seasonal,forecast_df_tes_Residual],axis=1)
   
    st=time.time()
    forecast_df=ARIMA_check(df_try['Trend'],forecast_length,forecast_df,'Trend')
    forecast_df=ARIMA_check(df_try['Residual'],forecast_length,forecast_df,'Residual')
    end=time.time()
    print("decompose_arima_trend",end-st)

    # st=time.time()
    # forecast_df=TBATS_s(df_try['Seasonal'],forecast_length,forecast_df)
    # end=time.time()
    # print("decompose_tbats",end-st)

    st=time.time()
    forecast_df=SARIMA_check(df_try['Seasonal'],forecast_length,forecast_df,'Seasonal')
    end=time.time()
    print("decompose_sarima_seasonal",end-st)
    
    st=time.time()
    forecast_df=ARMA_check(df_try['Residual'],forecast_length,forecast_df,'Residual')
    end=time.time()
    print("decompose_arma_resudial",end-st)
    
    print('decompos_model',forecast_df)
    return forecast_df

def pipeline3_forecast(train_check3,train_check_after_zeroes_r,test_check3,no_months_forecast):
    for fin_id in train_check3.unique_id.unique():
        try_check=train_check3[train_check3.unique_id==fin_id]
        try_check.set_index(['Date'],inplace=True)
        first_index=removal_zeroes(try_check['Values'])
        try_check_n=try_check[first_index:]
        try_check_n.reset_index(inplace=True)
        train_check_after_zeroes_r=pd.concat([train_check_after_zeroes_r,try_check_n],axis=0)
    train_check_after_zeroes_r=train_check3
    test_post_convert=test_check3.copy()
    train_sd=pd.DataFrame({})
    test_sd=pd.DataFrame({})
    st=time.time()
    print('train_check_after_zeroes_r',train_check_after_zeroes_r)
    train_sd=decomposition(train_check_after_zeroes_r,train_sd)
    print('train_sd',train_sd)
    end=time.time()
    print(f'time taken to run stl decomposition {end-st}')
    
    i=3
    seasonal_count=[]
    while(i<12):
        i+=1
        seasonal_count.append(i)
    seasonal_period= seasonal_count 
    # seasonal_period=[8,9,10,11,12]
    initial_method='estimated'
    master_forecast_df=pd.DataFrame()
    for fin_id in train_sd.unique_id.unique(): 
        # if fin_id in ["FOODS_1_015_TX_2","FOODS_1_015_WI_2"]:
        #     seasonal_period=[8]
        # elif fin_id in ["FOODS_1_001_WI_3","FOODS_1_011_TX_2"]:
        #     seasonal_period=[9]
        # elif fin_id in ["FOODS_1_004_CA_2"]:
        #     seasonal_period=[10]
        # elif fin_id in ["FOODS_1_015_CA_1"]:
        #     seasonal_period=[11]
        # else:
        #     seasonal_period=[12]
        df_try=train_sd[train_sd.unique_id==fin_id].copy()
        test_df_try=test_post_convert[test_post_convert.unique_id==fin_id]
        forecast_length=len(test_df_try) + no_months_forecast
        df_try.set_index(['Date'],inplace=True)
        fft_result=fft(np.array(df_try['Values']))
        n=len(df_try)
        freq=np.fft.fftfreq(n)
        peaks = np.where(np.abs(fft_result) > 0.2 * np.max(np.abs(fft_result)))
        periods = 1 / freq[peaks]
        
        st=time.time()
        print('df_try',df_try)
        forecast_df=decompose(df_try,forecast_length,seasonal_period,initial_method)
        end=time.time()
        print(f'time taken to run decompose model in pipeline3 {end-st}')

        forecast_df['Intermittency_check']=test_df_try['Intermittency_Type'].values[-1]
        forecast_df['unique_id']=fin_id
        forecast_df['Actual'] = np.nan
        n=test_df_try.shape[0]
        forecast_df.loc[:n, 'Actual'] = test_df_try['Values'].reset_index(drop=True)
        forecast_df=forecast_df.reset_index()
        df=test_df_try[['Date']]
        last_date = test_df_try['Date'].iloc[-1]
        extended_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=no_months_forecast, freq='MS')
        extended_df = pd.DataFrame({'Date': extended_dates})
        result_df = pd.concat([df, extended_df], ignore_index=True)
        forecast_df['Date']=result_df['Date']
        # forecast_df=forecast_df.drop('index')
        master_forecast_df=pd.concat([master_forecast_df,forecast_df],axis=0)
        # print('master_forecast_df',master_forecast_df)
        # master_forecast_df.to_csv('checkchekc.csv')
        test_df_try.reset_index(inplace=True)
    # master_forecast_df['Resultant1']=master_forecast_df['TES_Trend']+master_forecast_df['SARIMA_Seasonal']+master_forecast_df['ARMA_Residual']
    # master_forecast_df['Resultant2']=master_forecast_df['ARIMA_Trend']+master_forecast_df['SARIMA_Seasonal']+master_forecast_df['ARMA_Residual']
    # master_forecast_df['Resultant3']=master_forecast_df['TES_Trend']+master_forecast_df['SARIMA_Seasonal']+master_forecast_df['ARIMA_Residual']
    # master_forecast_df['Resultant4']=master_forecast_df['TES_Trend']+master_forecast_df['SARIMA_Seasonal']+master_forecast_df['TES_Residual']
    # master_forecast_df['Resultant5']=master_forecast_df['TES_Trend']+master_forecast_df['TES_Seasonal']+master_forecast_df['TES_Residual']
    # master_forecast_df['Resultant6']=master_forecast_df['TES_Trend']+master_forecast_df['TES_Seasonal']+master_forecast_df['ARMA_Residual']
    # master_forecast_df['Resultant7']=master_forecast_df['ARIMA_Trend']+master_forecast_df['TES_Seasonal']+master_forecast_df['ARMA_Residual']
    # master_forecast_df['Resultant8']=master_forecast_df['TES_Trend']+master_forecast_df['TES_Seasonal']+master_forecast_df['ARIMA_Residual']
    
    required_columns1 = ['SARIMA_Seasonal','TES_Seasonal','TES_Trend','TES_Residual','ARMA_Residual','ARIMA_Trend','ARIMA_Residual']
    required_columns2 = ['ARIMA_Trend', 'SARIMA_Seasonal', 'ARIMA_Trend', 'ARMA_Residual']

    if all(col in master_forecast_df.columns for col in required_columns1):
        master_forecast_df['Resultant1'] = master_forecast_df['TES_Trend'] + master_forecast_df['SARIMA_Seasonal'] + master_forecast_df['ARMA_Residual']
        master_forecast_df['Resultant2'] = master_forecast_df['TES_Trend'] + master_forecast_df['SARIMA_Seasonal'] + master_forecast_df['ARIMA_Residual']
        master_forecast_df['Resultant3'] = master_forecast_df['TES_Trend'] + master_forecast_df['SARIMA_Seasonal'] + master_forecast_df['TES_Residual']
        master_forecast_df['Resultant4'] = master_forecast_df['ARIMA_Trend'] + master_forecast_df['SARIMA_Seasonal'] + master_forecast_df['TES_Residual']
        master_forecast_df['Resultant5'] = master_forecast_df['TES_Trend'] + master_forecast_df['TES_Seasonal'] + master_forecast_df['ARMA_Residual']
        master_forecast_df['Resultant6'] = master_forecast_df['TES_Trend'] + master_forecast_df['TES_Seasonal'] + master_forecast_df['ARIMA_Residual']
        master_forecast_df['Resultant7'] = master_forecast_df['TES_Trend'] + master_forecast_df['TES_Seasonal'] + master_forecast_df['TES_Residual']
        master_forecast_df['Resultant8'] = master_forecast_df['ARIMA_Trend'] + master_forecast_df['TES_Seasonal'] + master_forecast_df['TES_Residual']
        master_forecast_df['Resultant9'] = master_forecast_df['ARIMA_Trend'] + master_forecast_df['TES_Seasonal'] + master_forecast_df['ARMA_Residual']
        master_forecast_df['Resultant10'] = master_forecast_df['ARIMA_Trend'] + master_forecast_df['SARIMA_Seasonal'] + master_forecast_df['ARMA_Residual']
        
        master_forecast_df['Resultant11'] = master_forecast_df['TES_Trend'] + master_forecast_df['SARIMA_Seasonal'] 
        master_forecast_df['Resultant12'] = master_forecast_df['ARIMA_Trend'] + master_forecast_df['SARIMA_Seasonal'] 
        master_forecast_df['Resultant13'] = master_forecast_df['TES_Trend'] + master_forecast_df['TES_Seasonal'] 
        master_forecast_df['Resultant14'] = master_forecast_df['ARIMA_Trend'] + master_forecast_df['TES_Seasonal'] 
        
    
        
    elif all(col in master_forecast_df.columns for col in required_columns2):
        master_forecast_df['Resultant10'] = master_forecast_df['ARIMA_Trend'] + master_forecast_df['SARIMA_Seasonal'] + master_forecast_df['ARMA_Residual']
        master_forecast_df['Resultant12'] = master_forecast_df['ARIMA_Trend'] + master_forecast_df['SARIMA_Seasonal']
        
    else:
        print("One or more required columns are not present in the DataFrame.")
    
    return train_check_after_zeroes_r,master_forecast_df,test_post_convert,train_sd

def pipeline4_forecast(train_check3,train_check_after_zeroes_r,test_post_convert,temp_final_check,no_months_forecast):
    for fin_id in train_check3.unique_id.unique():
        train_check_temp=train_check_after_zeroes_r[train_check_after_zeroes_r.unique_id==fin_id]
        test_df_try=test_post_convert[test_post_convert.unique_id==fin_id]
        forecast_df=pd.DataFrame()
        forecast_length=len(test_df_try) + no_months_forecast
        train_check_temp.set_index(['Date'],inplace=True)

        initial_method='estimated'
        seasonal_period=[3,4,5,6,7,8,9,10,11,12]
        # st=time.time()
        # forecast_df=TBATS_s(train_check_temp['Values'],forecast_length,forecast_df)
        # end=time.time()
        # print(f"time for running TBAT_s is {end-st}")

        # train_check_temp.reset_index(inplace=True)
        # st=time.time()
        # forecast_df_croston=CROSTON_f(train_check_temp[['unique_id','Date','Values']],forecast_length,pd.DataFrame())

        # end=time.time()
        # print(f"time for running CROSTON_f is {end-st}")

        # st=time.time()
        # forecast_df=NBEATS_model(train_check_temp[['unique_id','Date','Values']],forecast_length,forecast_df,200)
        # end=time.time()
        # print(f"time for running NBEATS_model is {end-st}")
        st_tes=time.time()
        # forecast_df_tes=TES(train_check_temp['Values'],forecast_length,forecast_df,seasonal_period,initial_method,'')
        forecast_df_TES=TES(train_check_temp['Values'],forecast_length,pd.DataFrame(),seasonal_period,initial_method,'')
      
        end_tes=time.time()
        print("TES time",end_tes-st_tes)
        
        # forecast_df=pd.concat([forecast_df_croston,forecast_df_TES],axis=1)
        forecast_df=forecast_df_TES
        print("combine_forcast df",forecast_df)
        
        forecast_df['Intermittency_check']=test_df_try['Intermittency_Type'].values[-1]
        forecast_df['unique_id']=fin_id
        forecast_df['Actual'] = np.nan
        n=test_df_try.shape[0]
        forecast_df.loc[:n, 'Actual'] = test_df_try['Values'].reset_index(drop=True)
        forecast_df=forecast_df.reset_index()
        
        df=test_df_try[['Date']]
        last_date = test_df_try['Date'].iloc[-1]
        extended_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=no_months_forecast, freq='MS')
        extended_df = pd.DataFrame({'Date': extended_dates})
        result_df = pd.concat([df, extended_df], ignore_index=True)
        forecast_df['Date']=result_df['Date']

        train_check_temp.reset_index(inplace=True)
        test_df_try.reset_index(inplace=True)
        temp_final_check=pd.concat([temp_final_check,forecast_df],axis=0)
    return train_check_after_zeroes_r,temp_final_check



def metrics_evaluation(forecast_df,models,train_try,fin_id):
    # best_mape=float('inf')
    best_mase=float('inf')
    best_cosine_similarity =0
    best_dtw = float('inf')
    best_smape = float('inf')
    # best_combined_metric = float('inf')
    # best_cosine=0
    print('*')
    best_model_mase='mase'
    best_model_cosine='cosine'
    best_model_dtw='dtw'
    best_model_smape='smape'
    # combined_metric_best_model='combine_model'
    # print('forecast_df',forecast_df)
    forecast_df = forecast_df.dropna(subset=['Actual'])
    train_try = train_try.dropna(subset=['Values'])
    print('forecast_df',forecast_df)
    print('train_try',train_try)
    # w1=0.33
    # w2=0.33
    # w3=0.33
    
    models1=[]
    for col in forecast_df.columns:
        if col in models:
            models1.append(col)
            
    mase_dic = {}
    smape_dic = {}
    best_model_dic={}
    if forecast_df['Intermittency_check'].isin(['Lumpy','Intermittent']).all():
        for model in models1:
            print('model',model)
            try:
                mase=mean_absolute_scaled_error(forecast_df['Actual'],forecast_df[model],y_train=train_try['Values'].values,multioutput='raw_values')[0]
                mase_dic[model]=mase
                
                smape_list=[]
                for index, row in forecast_df.iterrows():
                    if row[model]==0 and row['Actual']==0:
                        smape_list.append(0)
                    else:
                        smape_list.append((np.abs(row['Actual']-row[model]) / (np.abs(row['Actual'])+np.abs(row[model]))*0.5 ))
                smape = np.mean(smape_list)
                smape_dic[model]=smape
                
                actual_normalized = MinMaxScaler().fit_transform(forecast_df['Actual'].values.reshape(-1, 1)).flatten()
                forecast_normalized = MinMaxScaler().fit_transform(forecast_df[model].values.reshape(-1, 1)).flatten()
                pattern_similarity = (1 + cosine(actual_normalized, forecast_normalized)) / 2
                # pattern_similarity = (1+cosine(forecast_df['Actual'],forecast_df[model]))/2

                _, path = fastdtw(actual_normalized, forecast_normalized)
                dtw_distance = len(path)
                # combined_metric = 1/(w1 *(1/(1+mase)) + w2 * pattern_similarity+w3*(1/(1+dtw_distance)))
                
                # if combined_metric <best_combined_metric:
                #     combined_metric_best_model=model
                #     best_combined_metric=combined_metric
                
                # print(f"{model} mase",mase)
                
                if mase <best_mase:
                    best_model_mase=model
                    best_mase=mase
                    
                if smape <best_smape:
                    best_model_smape=model
                    best_smape=smape
                    
                if dtw_distance <best_dtw:
                    best_model_dtw=model
                    best_dtw=dtw_distance
                    
                if pattern_similarity > best_cosine_similarity:
                    best_model_cosine=model
                    best_cosine_similarity=pattern_similarity
            except Exception as e:
                print(e)
                pass
        # print("lumpy & intermittent",best_mase,best_model)
        best_models=[best_model_mase,best_model_smape,best_model_dtw,best_model_cosine]
        print('best_models',best_models)
        for i in best_models:
            if i in best_model_dic.keys():
                best_model_dic[i]+=1
            else:
                best_model_dic[i]=1      
        sorted_dict_desc = dict(sorted(best_model_dic.items(), key=lambda item: item[1], reverse=True))
        print('fin_id',fin_id)
        print('sorted_dict_desc',sorted_dict_desc)
        smape_model=next(iter(sorted_dict_desc))

        if sorted_dict_desc[smape_model]>1 :
            return smape_dic[smape_model],smape_model
        else :
            return smape_dic[best_model_cosine],best_model_cosine
    else:   
        for model in models1:
            print('***')
            try:
                actual_normalized = MinMaxScaler().fit_transform(forecast_df['Actual'].values.reshape(-1, 1)).flatten()
                forecast_normalized = MinMaxScaler().fit_transform(forecast_df[model].values.reshape(-1, 1)).flatten()
                pattern_similarity = (1 + cosine(actual_normalized, forecast_normalized)) / 2
                forecast_df[model+'_Error'] = forecast_df['Actual'] - forecast_df[model]
                forecast_df[model+'_Absolute_Error'] = np.abs(forecast_df[model+'_Error'])
                # forecast_df.to_csv("forecast_df_error.csv")

                    
                # np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                # mape = np.mean(np.abs(actual_new[model+'_Absolute_Error'] / actual_new['Actual'])) * 100
                mase=mean_absolute_scaled_error(forecast_df['Actual'],forecast_df[model],y_train=train_try['Values'].values,multioutput='raw_values')[0]
                mase_dic[model]=mase
                
                smape_list=[]
                for index, row in forecast_df.iterrows():
                    if row[model]==0 and row['Actual']==0:
                        smape_list.append(0)
                    else:
                        smape_list.append(np.abs(row[model+'_Absolute_Error'] / (np.abs(row['Actual'])+np.abs(row[model]))*0.5 ))
                smape = np.mean(smape_list)
                smape_dic[model]=smape
                
                _, path = fastdtw(actual_normalized, forecast_normalized)
                dtw_distance = len(path)
                
                # combined_metric = 1/(w1 *(1/(1+mape)) + w2 * pattern_similarity+w3*(1/(1+dtw_distance)))
                
                # if combined_metric <best_combined_metric:
                #     combined_metric_best_model=model
                #     best_combined_metric=combined_metric
                    
                if mase <best_mase:
                    best_model_mase=model
                    best_mase=mase
                    
                if smape <best_smape:
                    best_model_smape=model
                    best_smape=smape
                    
                if dtw_distance <best_dtw:
                    best_model_dtw=model
                    best_dtw=dtw_distance
                    
                if pattern_similarity > best_cosine_similarity:
                    best_model_cosine=model
                    best_cosine_similarity=pattern_similarity
            except Exception as e:
                print(e)
                pass
    
        best_models=[best_model_mase,best_model_smape,best_model_dtw,best_model_cosine]
        print('best_models',best_models)
        for i in best_models:
            if i in best_model_dic.keys():
                best_model_dic[i]+=1
            else:
                best_model_dic[i]=1
                
        sorted_dict_desc = dict(sorted(best_model_dic.items(), key=lambda item: item[1], reverse=True))
        smape_model=next(iter(sorted_dict_desc))
        print('fin_id',fin_id)
        print('sorted_dict_desc',sorted_dict_desc)
        if sorted_dict_desc[smape_model]>1 :
            return smape_dic[smape_model],smape_model
        else :
            return smape_dic[best_model_cosine],best_model_cosine
        # return best_mape,best_model

def train_predict(data,no_months_forecast):

    required_col_name=["unique_id", "Date", "Values"]
    data.columns=required_col_name
    id_counts = data['unique_id'].value_counts()
    selected_ids = id_counts[id_counts >= 30].index.tolist()
    data = data[data['unique_id'].isin(selected_ids)]
    # print("data id's more than 30 or 30",data['unique_id'])
    print("data",data.head())
    st=time.time()
    percent_split=20
    train,test=train_test_split(data,percent_split)
    end=time.time()
    print("train_test_split time",end-st)
    
    # creating intermittency_dict
    st=time.time()
    intermittency_dict={}
    for m_id in train.unique_id:
        check_df=train[train.unique_id==m_id]
        check_df=check_df.pivot(columns='unique_id',values='Values')
        label=intermittency_dict_label(check_df,m_id)
        if label in intermittency_dict:
            if m_id not in intermittency_dict[label]:
                intermittency_dict[label].append(m_id)
            else:
                continue
        else:
            intermittency_dict[label] = [m_id]
    end=time.time()
    print("intermittency_dict_label_time",end-st)

    # creating intermittency_list
    intermittency_list=[]
    for label,v in intermittency_dict.items():
        intermittency_dict.update({label:v[:]})
        for item in v[:]:
            intermittency_list.append(item)

    st=time.time()
    test_check,train_check=full_train_test(train,test,intermittency_dict)
    end=time.time()
    print("full_train_test_time",end-st)
    
    print("intermittency_dict",intermittency_dict)

    #converting date into datetime object
    train_check['Date']=pd.to_datetime(train_check['Date'])
    test_check['Date']=pd.to_datetime(test_check['Date'])

    # dividing data according to pipelines
    train_check1=train_check[train_check['Intermittency_Type'].isin(['Smooth','Erratic'])]
    test_check1=test_check[test_check['Intermittency_Type'].isin(['Smooth','Erratic'])]
    
    train_check3=train_check[train_check['Intermittency_Type'].isin(['Lumpy','Intermittent'])]
    test_check3=test_check[test_check['Intermittency_Type'].isin(['Lumpy','Intermittent'])]
    
    # if outlier_removal=="True":
    #     # print("outlier_removal",outlier_removal)
    #     # removal of outliers
    #     train['Values'] = hampel_filter(train['Values'], 3, 3)
    #     test['Values'] = hampel_filter(test['Values'], 3, 3)
    


    isin_pipeline_1,isin_pipeline_2=False,False

    # for running pipeline 1
    if len(train_check1)>0:
        st=time.time()
        isin_pipeline_1=True
        # models=['ARMA','SARIMA','TES','SES','DES','ARIMA','SKFORECAST_HISTGRADBOOST','SKFORECAST_CATBOOST','SKFORECAST_LGB','SKFORECAST_XGB']
        # seasonal_period=[3,4,5,6,7,8,9,10,11,12]
        initial_method='heuristic'
        train_check1['Values']=train_check1['Values'].astype(float).round(2)

        master_forecast_df=pd.DataFrame()
        for fin_id in list(train_check1['unique_id'].unique()):
            if fin_id in ["FOODS_1_001_WI_3","FOODS_1_011_TX_2"]:
                seasonal_period=[9]
            elif fin_id in ["FOODS_1_004_CA_2"]:
                seasonal_period=[10]
            else:
                seasonal_period=[3,4,5,6,7,8,9,10,11,12]
            forecast_df=pipeline1_forecast (train_check1,test_check1,fin_id,no_months_forecast,seasonal_period,initial_method)
            master_forecast_df=pd.concat([master_forecast_df,forecast_df],axis=0)
        end=time.time()
        print("pipeline 1 running time",end-st)
        
        # for running pipeline 2
        st=time.time()
        master_forecast_df_2=pd.DataFrame()
        params_struct=pd.DataFrame()
        start_time=time.time()

        for fin_id in list(train_check1['unique_id'].unique()):
            print('fin_id',fin_id)
            forecast_df=pipeline2_forecast(train_check1,test_check,master_forecast_df,fin_id,params_struct,no_months_forecast)
            master_forecast_df_2=pd.concat([master_forecast_df_2,forecast_df],axis=0)
        # master_forecast_df_2.to_csv(os.path.join(output_dir,'forecast_results2'+'.csv'),index=False)  
        end=time.time()
        print("pipeline 2 running time",end-st)
        master_forecast_df.reset_index(drop=True, inplace=True)
        master_forecast_df_2.reset_index(drop=True, inplace=True)
        print('master_forecast_df_2',master_forecast_df_2)
        print('master_forecast_df',master_forecast_df)
        # combining pipeline_1 and pipeline_2 resultsl
        master_forecast_df_fin=pd.concat([master_forecast_df,master_forecast_df_2],axis=1)
        
        # master_forecast_df_fin=master_forecast_df
        print("master_forecast_df_fin",master_forecast_df_fin)
        # to remove the duplicate values from the dataframe
        master_forecast_df_fin = master_forecast_df_fin.loc[:, ~master_forecast_df_fin.columns.duplicated()]
        master_forecast_df_fin.to_csv('pipline1&2.csv')
        
        exclude=['unique_id','Actual','Date','Intermittency_check']
        models=[col for col in master_forecast_df_fin.columns if col not in exclude]
        print('models_pipline1_2',models)
        
        
        # print("after pipeline 1&2",master_forecast_df_fin.head())
        # for calculating the metric evaluation and best models
        final_error_metric_df1=pd.DataFrame(columns=['unique_id','Intermittency_Type','Best_Model_Evaluated','MAPE'])
        for fin_id in list(train_check1['unique_id'].unique()):
            try:
                print("fin_id",fin_id)
                train_try=train[train['unique_id']==fin_id]
                train_try.reset_index(drop=True,inplace=True)
                forecast_df=master_forecast_df_fin[master_forecast_df_fin['unique_id']==fin_id]
                forecast_df.reset_index(drop=True,inplace=True)
                # print("error_check",forecast_df)
                # forecast_df.to_csv("error_check.csv")
                best_mape,best_model=metrics_evaluation(forecast_df,models,train_try,fin_id)
                print("best_mape,best_model",best_mape,best_model)
                if best_model!='XYZ':
                    new_df=pd.DataFrame({'unique_id':fin_id,'Intermittency_Type':str(forecast_df['Intermittency_check'][0]),'Best_Model_Evaluated':best_model,'MAPE':best_mape},index=[0])
                    final_error_metric_df1=pd.concat([final_error_metric_df1,new_df],axis=0)
            except Exception as e:
                print(e)
                pass
        if len(final_error_metric_df1)>0:
            print("final_error_metric_df1",final_error_metric_df1.head())
            final_ormaefit_output=pd.DataFrame()
            for fin_id in final_error_metric_df1.unique_id.unique():
                mid_frame=pd.DataFrame()
                forecast_df=master_forecast_df_fin[master_forecast_df_fin['unique_id']==fin_id]
                best_model=final_error_metric_df1[final_error_metric_df1.unique_id==fin_id]['Best_Model_Evaluated'].values[0]
                mid_frame['Forecast']=forecast_df[best_model].astype(float).round(2)
                mid_frame['Actual']=forecast_df['Actual']
                mid_frame['unique_id']=fin_id
                mid_frame['Intermittency_Type']=forecast_df['Intermittency_check']
                mid_frame['Best_Model']=best_model
                mid_frame['Date']=forecast_df['Date']
                final_ormaefit_output=pd.concat([final_ormaefit_output,mid_frame],axis=0) 
            print("final_ormaefit_output",final_ormaefit_output.head()) 
            final_ormaefit_output = final_ormaefit_output[['Date','unique_id','Intermittency_Type','Best_Model','Actual','Forecast']]
            
            train_result=train_check1
            train_result.rename(columns={'Values':'Actual'},inplace=True)
            test_result = final_ormaefit_output[final_ormaefit_output['Actual'].notnull()]
            forecast_result = final_ormaefit_output[final_ormaefit_output['Actual'].isnull()]
            

            # assigning labels to train,test,valid
            train_result['label']='train'
            test_result['label']='test'
            forecast_result['label']='forecast'

            final_dataframe=pd.DataFrame()
            for id in list(test_result['unique_id'].unique()):
                try:
                    train_result1=train_result[train_result['unique_id']==id]
                    test_result1=test_result[test_result['unique_id']==id]
                    forecast_result1=forecast_result[forecast_result['unique_id']==id]
                    result1=calculate_prediction_interval(pd.concat([test_result1,forecast_result1]))
                    final_dataframe_id=pd.concat([train_result1,result1],axis=0,ignore_index=True)
                    final_dataframe=pd.concat([final_dataframe,final_dataframe_id])
                except Exception  as e:
                    print(f"An error occurred in {id}: {e}")
            final_dataframe1=final_dataframe[['unique_id','Date','Intermittency_Type','label','Best_Model','Actual','Forecast_Lower','Forecast','Forecast_Upper']]
            print("final_dataframe1",final_dataframe1)
        else: 
            isin_pipeline_1=False 
            
             
    if len(train_check3)>0:
        isin_pipeline_2=True
        start_time=time.time()
        train_check_after_zeroes_r=pd.DataFrame({})
        train_check_after_zeroes_r,master_forecast_df,test_post_convert,train_sd=pipeline3_forecast(train_check3,train_check_after_zeroes_r,test_check3,no_months_forecast)
        end_time=time.time()
        print("pipeline 3 running time",end_time-start_time)
        

        start_time=time.time()
        temp_final_check=pd.DataFrame({})
        train_check_after_zeroes_r,temp_final_check=pipeline4_forecast(train_check3,train_check_after_zeroes_r,test_post_convert,temp_final_check,no_months_forecast)
            
        end_time=time.time()
        print("pipeline 4 running time",end_time-start_time)
        
        master_forecast_df.drop('index',axis='columns', inplace=True)
        temp_final_check.drop('index',axis='columns', inplace=True)
        master_forecast_df.to_csv('pipeline3.csv')
        temp_final_check.to_csv('pipeline4.csv')
        pipeline_3_4_forcast_df=pd.concat([master_forecast_df,temp_final_check],axis=1)
        pipeline_3_4_forcast_df = pipeline_3_4_forcast_df.loc[:, ~pipeline_3_4_forcast_df.columns.duplicated()]
        pipeline_3_4_forcast_df.to_csv('pipline3&4.csv')

        models_int=['TES','ADIDA', 'CrostonClassic', 'IMAPA', 'TSB','CrostonOptimised', 'RWD', 'CrostonSBA']#,'NBEATS','NHITS']
        models_dec=['Resultant1','Resultant2','Resultant3','Resultant4','Resultant5','Resultant6','Resultant7','Resultant8','Resultant9','Resultant10']
        model_int_dec=models_int+models_dec
        final_error_metric_df2=pd.DataFrame()
        Fin_unique_ids=train_check3.unique_id.unique()
        for fin_id in Fin_unique_ids:
            try:
                forecast_df=pipeline_3_4_forcast_df[pipeline_3_4_forcast_df['unique_id']==fin_id]
                df_try=train_check_after_zeroes_r[train_check_after_zeroes_r.unique_id==fin_id]
                # df_try=train_sd[train_sd['unique_id']==fin_id]
                print("*")
                best_mase,best_model=metrics_evaluation(forecast_df,model_int_dec,df_try,fin_id)
                if best_model != 'XYZ':
                    final_error_metric_df2=final_error_metric_df2._append({'unique_id':fin_id,'Intermittency_Type':str(forecast_df['Intermittency_check'][0]),'Best_Model_Evaluated':best_model,'MASE':best_mase},ignore_index=True)
            except Exception as e:
                print(e)
                pass
            
        print("final_error_metric_df2",final_error_metric_df2)
        print()
        
        inter_res=final_error_metric_df2
        if len(final_error_metric_df2)>1:
            best_model_df=pd.DataFrame({})
            for fin_id in inter_res.unique_id.unique():
                temp_df=pipeline_3_4_forcast_df[pipeline_3_4_forcast_df.unique_id==fin_id]
                temp_df_1=temp_df[['Date','Intermittency_check','unique_id','Actual']]
                temp_df_1['Best_Model']=inter_res[inter_res.unique_id==fin_id]['Best_Model_Evaluated'].values[0]
                temp_df_1['Best_MASE']=inter_res[inter_res.unique_id==fin_id]['MASE'].values[0]
                b_model=inter_res[inter_res.unique_id==fin_id]['Best_Model_Evaluated'].values[0]
                temp_df_1['Forecast']=temp_df[b_model].values.astype(float).round(2)
                best_model_df=pd.concat([best_model_df,temp_df_1],axis=0)
                del temp_df_1
                    
            train_result=train_check3.rename(columns={'Values':'Actual','Intermittency_Type':'Intermittency_check'})
            test_result = best_model_df[best_model_df['Actual'].notnull()]
            forecast_result = best_model_df[best_model_df['Actual'].isnull()]
            
            # assigning labels to train,test,valid
            train_result['label']='train'
            test_result['label']='test'
            forecast_result['label']='forecast'
            
            final_dataframe=pd.DataFrame()
            for id in list(test_result['unique_id'].unique()):
                train_result1=train_result[train_result['unique_id']==id]
                test_result1=test_result[test_result['unique_id']==id]
                forecast_result1=forecast_result[forecast_result['unique_id']==id]
                result1=calculate_prediction_interval(pd.concat([test_result1,forecast_result1]))
                final_dataframe_id=pd.concat([train_result1,result1],axis=0,ignore_index=True)
                final_dataframe=pd.concat([final_dataframe,final_dataframe_id])
            final_dataframe2=final_dataframe[['unique_id','Date','Intermittency_check','label','Best_Model','Actual','Forecast_Lower','Forecast','Forecast_Upper']]
            final_dataframe2.rename(columns={'Intermittency_check':'Intermittency_Type'},inplace=True)
        else:
            isin_pipeline_2=False
    
    # storing the metrics and the best models
    print("isin_pipeline_1,isin_pipeline_2",isin_pipeline_1,isin_pipeline_2)
    if isin_pipeline_1==True and isin_pipeline_2==True:
        final_error_metric_df=pd.concat([final_error_metric_df1,final_error_metric_df2],axis=0)
    elif isin_pipeline_2==True and isin_pipeline_1==False:
        final_error_metric_df=final_error_metric_df2
    elif isin_pipeline_2==False and isin_pipeline_1==True:
        final_error_metric_df=final_error_metric_df1
        
    final_error_metric_df.to_csv("metric_evaluation.csv")
    
    dff=pd.DataFrame()
    if isin_pipeline_2==True and isin_pipeline_1==False:
        dff= final_dataframe2
    elif isin_pipeline_1==True and isin_pipeline_2==False:
        dff= final_dataframe1
    elif isin_pipeline_1==True and isin_pipeline_2==True:
        dff=pd.concat([final_dataframe1,final_dataframe2],axis=0,ignore_index=True)
    else:
        return "Data Is Not Sufficient"
    dff[['Actual','Forecast_Lower','Forecast','Forecast_Upper']]=dff[['Actual','Forecast_Lower','Forecast','Forecast_Upper']].astype(float).round(2)
    # dff[dff['label'].isin(['test','forecast'])]['Forecast']=dff[dff['label'].isin(['test','forecast'])]['Forecast'].apply(lambda x: max(0, x))
    # dff.loc[dff['Forecast'] == 0, 'Forecast_Upper'] = 0
    mask = (dff['label'].isin(['test', 'forecast'])) & (dff['Forecast'] < 0) 
    dff.loc[mask, ['Forecast', 'Forecast_Lower', 'Forecast_Upper']] = 0
    # current_date = datetime.now().date()
    dff.to_csv("FOOD_SKUs_result_latest.csv")
    return dff

# path="/Workspace/Users/amit_ku@ormae.com/SKUs_85.csv"
# data=pd.read_csv(path)
# print(train_predict(data,12))