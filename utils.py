from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt


from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel, NBEATSModel
from darts.models.forecasting.xgboost import XGBModel
from darts.models.forecasting.rnn_model import RNNModel

from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, smape, mae
from darts.utils.utils import ModelMode

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)
 
def predict(df,country,steps):
    y = df[[country]]

    # Create trend features
    dp = DeterministicProcess(
        index=y.index,  # dates from the training data
        constant=True,  # the intercept
        order=1,        # quadratic trend
        drop=True,      # drop terms to avoid collinearity
    )
    X = dp.in_sample()  # features for the training data

    X_pred = dp.out_of_sample(steps=steps)


    # Fit trend model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Make predictions
    y_fit = pd.DataFrame(
        model.predict(X),
        index=y.index,
        columns=y.columns,
    )
    y_pred = pd.DataFrame(
        model.predict(X_pred),
        index=X_pred.index,
        columns=y.columns,
    )
    
    fit_df = pd.DataFrame(columns=[country],index=y_fit.index)
    fit_df.loc[y_fit.index,country] = y_fit.values.reshape(1,-1)
    
    pred_df = pd.DataFrame(columns=[country],index=y_pred.index)
    pred_df.loc[y_pred.index,country] = y_pred.values.reshape(1,-1)
    
    return fit_df,pred_df

def combine(hist_df,fits_df,fore_df):
    temp_hist_df = hist_df.copy().reset_index()
    temp_fits_df = fits_df.copy().reset_index()
    temp_fore_df = fore_df.copy().reset_index()
    temp_hist_df["Tag"] = "Historical"
    temp_fits_df["Tag"] = "Regression"
    temp_fore_df["Tag"] = "Forecast"
    return pd.concat([temp_hist_df,temp_fits_df,temp_fore_df],axis=0).rename(columns={"index":"Year"}).set_index("Year")

country_iso_alpha3 = {
    'Algeria': 'DZA',
    'Angola': 'AGO',
    'Benin': 'BEN',
    'Botswana': 'BWA',
    'Burkina Faso': 'BFA',
    'Burundi': 'BDI',
    'Cabo Verde': 'CPV',
    'Cameroon': 'CMR',
    'Central African Republic': 'CAF',
    'Chad': 'TCD',
    'Comoros': 'COM',
    'Congo': 'COG',
    "Côte d'Ivoire": 'CIV',
    'Democratic Republic of the Congo': 'COD',
    'Djibouti': 'DJI',
    'Egypt': 'EGY',
    'Equatorial Guinea': 'GNQ',
    'Eritrea': 'ERI',
    'Eswatini': 'SWZ',
    'Ethiopia': 'ETH',
    'Gabon': 'GAB',
    'Gambia': 'GMB',
    'Ghana': 'GHA',
    'Guinea': 'GIN',
    'Guinea-Bissau': 'GNB',
    'Kenya': 'KEN',
    'Lesotho': 'LSO',
    'Liberia': 'LBR',
    'Libya': 'LBY',
    'Madagascar': 'MDG',
    'Malawi': 'MWI',
    'Mali': 'MLI',
    'Mauritania': 'MRT',
    'Mauritius': 'MUS',
    'Morocco': 'MAR',
    'Mozambique': 'MOZ',
    'Namibia': 'NAM',
    'Niger': 'NER',
    'Nigeria': 'NGA',
    'Rwanda': 'RWA',
    'Sao Tome and Principe': 'STP',
    'Senegal': 'SEN',
    'Sierra Leone': 'SLE',
    'Somalia': 'SOM',
    'South Africa': 'ZAF',
    'South Sudan': 'SSD',
    'Sudan': 'SDN',
    'Tanzania': 'TZA',
    'Togo': 'TGO',
    'Tunisia': 'TUN',
    'Uganda': 'UGA',
    'Zambia': 'ZMB',
    'Zimbabwe': 'ZWE'
}


def es_model(data, country, step):
    
    
    es_model = ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=None)
    es_model.fit(data[country])

    es_pred = es_model.predict(step)

    return data[country], es_pred

