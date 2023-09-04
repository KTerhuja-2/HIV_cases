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
    
    pred_df = pd.DataFrame(columns=[country,"History"],index=y_pred.index.union(y.index))
    
    pred_df.loc[y_pred.index,country] = y_pred.values.reshape(1,-1)
    pred_df.loc[y_fit.index,country] = y_fit.values.reshape(1,-1)
    pred_df.loc[y.index,country] = y.values.reshape(1,-1)
    
    pred_df.loc[y_pred.index,"History"] = "Forecast"
    pred_df.loc[y_fit.index,"History"] = "Forecast"
    pred_df.loc[y.index,"History"] = "HIstorical"
    
    pred_df.rename(columns={country:"HIV Population"},inplace=True)
    pred_df.rename_axis("Years",inplace=True)
    return pred_df