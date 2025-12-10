# src/models/train_model.py
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    processed_train = home_dir / "data" / "processed" / "train.csv"
    df = pd.read_csv(processed_train)

    # NYC Taxi: target column is usually `trip_duration` (in seconds)
    target_col = "trip_duration"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    rmsle = mean_squared_log_error(y_valid, y_pred) ** 0.5
    print(f"Validation RMSLE: {rmsle:.4f}")

    model_path = home_dir / "model.joblib"
    joblib.dump({"model": model, "feature_cols": list(X.columns)}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
