from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

DATA_PATH = Path(__file__).resolve().parent / "kos_madiun.csv"
FITUR = ["jarak_km", "ac_num", "rating"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["ac_num"] = (df["ac"] == "ya").astype(int)  # ya=1, tidak=0
    return df


def latih_model(df):
    X, y = df[FITUR], df["harga"]
    model = LinearRegression()
    # Data hanya 30 baris, jadi dievaluasi dengan cross validation
    y_cv = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=42))
    mae = mean_absolute_error(y, y_cv)
    r2 = r2_score(y, y_cv)
    model.fit(X, y)
    return model, mae, r2


def prediksi(model, jarak, ac, rating):
    ac_num = 1 if ac == "ya" else 0
    X = pd.DataFrame([[jarak, ac_num, rating]], columns=FITUR)
    return float(model.predict(X)[0])


# Dijalankan sekali saat server mulai
DF = load_data()
MODEL, MAE, R2 = latih_model(DF)