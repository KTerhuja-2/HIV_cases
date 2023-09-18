import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

import pickle


from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel, NBEATSModel
from darts.models.forecasting.xgboost import XGBModel
from darts.models.forecasting.rnn_model import RNNModel

from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, smape, mae


df_africa = pd.read_csv("HIV_data 1990-2022.csv")

def load_data(path):

    df_africa = pd.read_csv(path)
    df_africa.rename(columns = {"Unnamed: 0": "Time"},inplace=True)
    df_africa.dropna(axis=1,inplace=True)
    df_africa['Time'] = pd.to_datetime(df_africa['Time'],format= "%Y")

    return df_africa

def create_time_series(dataframe):

    series = TimeSeries.from_dataframe(dataframe,
                                    time_col='Time'
                                    )
    return series

def split_data(time_series, date):
    train, val = time_series.split_before(pd.Timestamp(date))

    return train, val

def scale_and_inverse_scale(train, val, scaler=None):

    if scaler is None:
        scaler = Scaler()
        scaled_train = scaler.fit_transform(train)
        scaled_val = scaler.fit_transform(val)

        return train, val, scaler

    else:
        original_train = scaler.inverse_transform(train)
        original_val = scaler.inverse_transform(val)
        return train, val


def train_model(data, date):
    series = create_time_series(data)
    train, val = split_data(series, date=date)
    scaled_train, scaled_val, scaler = scale_and_inverse_scale(train, val)
    model = XGBModel(lags=5)
    model.fit([train])

    return model, scaler


def save_scaler(scaler, path):
    with open(path, 'wb') as handle:
        pickle.dump(scaler, handle)




if __name__ == "__main__":

    data = load_data('HIV_data 1990-2022.csv')
    
    trained_model, scaler = train_model(data, date="20150101")

    save_scaler(scaler, 'models/scaler.pkl')
    trained_model.save("models/xgboost.pkl")
    
    

