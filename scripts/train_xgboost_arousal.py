import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from app.ml.mood_mapping import (
    get_mood_mapping_version
)


# ---------------------------------
# Paths
# ---------------------------------

FEATURE_PATH = Path(
    "data/features/deam_features_v2.csv"
)

MODEL_OUTPUT = Path(
    "models/xgboost_arousal_v2.pkl"
)


print("\nLoading feature dataset...")

df = pd.read_csv(
    FEATURE_PATH
)

print(df.head())


# ---------------------------------
# Features / Labels
# ---------------------------------

X = df.drop(
    columns=[
        "song_id",
        "valence",
        "arousal"
    ]
)

y = df["arousal"]


print("\nFeature Shape:")
print(X.shape)


# ---------------------------------
# Train/Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ---------------------------------
# MLflow Tracking
# ---------------------------------

with mlflow.start_run(
    run_name="xgboost_arousal_v2"
):

    mlflow.log_param(
        "model_type",
        "XGBoost"
    )

    mlflow.log_param(
        "feature_version",
        "v2"
    )

    mlflow.log_param(
        "target",
        "arousal"
    )

    mlflow.log_param(
        "mood_mapping_version",
        get_mood_mapping_version()
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1
    )

    print("\nTraining XGBoost...")

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
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_OUTPUT
    )

    mlflow.sklearn.log_model(
        model,
        artifact_path="xgboost_arousal_model"
    )

    print("\nTraining Complete")
    print("--------------------------")

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    print("\nSaved model to:")
    print(MODEL_OUTPUT)
