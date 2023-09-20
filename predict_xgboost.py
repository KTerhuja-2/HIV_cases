import pickle
from darts.models import NBEATSModel
from darts.models.forecasting.xgboost import XGBModel

def load_model(path):
    model_loaded = XGBModel.load(path)
    return model_loaded

def load_scaler(path):
    with open(path, 'rb') as handle:
        scaler = pickle.load(handle)
        return scaler



def forecast(model_path, scaler_path, step=4):
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    forecasted = model.predict(step)
    scaled_forecast = scaler.inverse_transform(forecasted)
    return scaled_forecast

    

def predict(country_name, input_years):

    output = forecast(model_path = 'models/xgboost.pkl',
            scaler_path = 'models/scaler.pkl',
            step = input_years)
    
    return output[country_name]








