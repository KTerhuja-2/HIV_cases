from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd
 
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
    return pd.concat([temp_hist_df,temp_fore_df],axis=0).rename(columns={"index":"Year"}).set_index("Year")