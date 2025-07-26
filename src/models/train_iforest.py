import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from src.features.build_features import build_features

def train_isolation_forest(data_path, model_out_path):
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    X, preprocessor = build_features(df)

    model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    model.fit(X)

    joblib.dump({"model": model, "preprocessor": preprocessor}, model_out_path)
    print(f"Model saved to: {model_out_path}")

if __name__ == "__main__":
    train_isolation_forest("data/raw/warehouse_logs.csv", "models/iforest_model.pkl")
