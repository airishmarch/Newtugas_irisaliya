import joblib
import pandas as pd
from preprocessing.preprocessing import preprocess_data


def predict():
    df = preprocess_data()

    model = joblib.load("artifacts/model_agglomerative.pkl")

    X = df[['productivity','emissions_scaled','renewable_supply']]
    df['prediction'] = model.fit_predict(X)

    print(df[['Reference area','prediction']].head())

    return df


if __name__ == "__main__":
    predict()