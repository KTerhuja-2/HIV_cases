from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd
 
def predict(df,country,steps):
    y = pd.DataFrame(df[country]).copy()

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
    return y_pred