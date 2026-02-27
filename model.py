import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

df = pd.read_csv("bahrain_2021_2024_combined.csv")

if df.empty:
    raise ValueError("Dataset is empty.")

df = df.dropna()

if "TrackStatus" in df.columns:
    df = df[df["TrackStatus"].astype(str).str.contains("1")]

if df.empty:
    raise ValueError("No data left after filtering TrackStatus.")

y = df["LapTimeSeconds"]

X = df[[
    "Year",
    "LapNumber",
    "Stint",
    "TyreLife",
    "FreshTyre",
    "Position",
    "Compound",
    "Team",
    "Driver"
]]

X = pd.get_dummies(
    X,
    columns=["Compound", "Team", "Driver"],
    drop_first=True
)

feature_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

hybrid_model = VotingRegressor(
    estimators=[("rf", rf), ("xgb", xgb)]
)

hybrid_model.fit(X_train, y_train)

y_pred = hybrid_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Hybrid Model Performance")
print("RMSE:", round(rmse, 3))
print("MAE:", round(mae, 3))
print("R2:", round(r2, 3))

joblib.dump(hybrid_model, "bahrain_hybrid_model.pkl")
joblib.dump(feature_columns, "bahrain_model_columns.pkl")

print("Hybrid model saved successfully.")