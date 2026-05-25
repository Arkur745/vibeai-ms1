import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

from app.ml.mood_mapping import (
    get_mood_mapping_version
)


# ---------------------------------
# Paths
# ---------------------------------

DATA_PATH = Path(
    "data/features/deam_features_v2.csv"
)

MODEL_OUTPUT = Path(
    "models/random_forest_valence_v2.pkl"
)


def main():

    print("\nLoading feature dataset...")

    df = pd.read_csv(DATA_PATH)

    print(df.head())

    # ---------------------------------
    # Features and Labels
    # ---------------------------------

    X = df.drop(
        columns=[
            "song_id",
            "valence",
            "arousal"
        ]
    )

    y = df["valence"]

    print("\nFeature Shape:")
    print(X.shape)

    # ---------------------------------
    # Train Test Split
    # ---------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ---------------------------------
    # MLflow Tracking
    # ---------------------------------

    with mlflow.start_run(
        run_name="random_forest_valence_v2"
    ):

        mlflow.log_param(
            "mood_mapping_version",
            get_mood_mapping_version()
        )

        mlflow.log_param(
            "feature_version",
            "v2"
        )

        mlflow.log_param(
            "model_type",
            "RandomForestRegressor"
        )

        mlflow.log_param(
            "target",
            "valence"
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "max_depth",
            12
        )

        # ---------------------------------
        # Model
        # ---------------------------------

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        print("\nTraining RandomForest...")

        model.fit(
            X_train,
            y_train
        )

        # ---------------------------------
        # Predictions
        # ---------------------------------

        predictions = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # ---------------------------------
        # MLflow Metrics
        # ---------------------------------

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "MSE",
            mse
        )

        mlflow.log_metric(
            "R2",
            r2
        )

        # ---------------------------------
        # Save Model
        # ---------------------------------

        MODEL_OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            MODEL_OUTPUT
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="random_forest_valence_v2_model"
        )

        print("\nTraining Complete")
        print("--------------------------")

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")

        print("\nSaved model to:")
        print(MODEL_OUTPUT)


if __name__ == "__main__":
    main()
