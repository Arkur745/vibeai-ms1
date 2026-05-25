import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------
# Paths
# ---------------------------------

DATA_PATH = Path(
    "data/features/gtzan_features_v1.csv"
)

MODEL_OUTPUT = Path(
    "models/genre_classifier_v1.pkl"
)


def main():

    print("\nLoading genre feature dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(df.head())

    # ---------------------------------
    # Features / Labels
    # ---------------------------------

    X = df.drop(
        columns=["genre"]
    )

    y = df["genre"]

    print("\nFeature Shape:")
    print(X.shape)

    print("\nGenres:")
    print(y.unique())

    # ---------------------------------
    # Train Test Split
    # ---------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")

    print(f"Testing samples: {len(X_test)}")

    # ---------------------------------
    # MLflow Tracking
    # ---------------------------------

    with mlflow.start_run(
        run_name="genre_classifier_rf_v1"
    ):

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "feature_version",
            "v1"
        )

        # ---------------------------------
        # Model
        # ---------------------------------

        model = RandomForestClassifier(

            n_estimators=300,

            max_depth=15,

            random_state=42,

            n_jobs=-1
        )

        print("\nTraining genre classifier...")

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

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        # ---------------------------------
        # MLflow Metrics
        # ---------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
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
            artifact_path="genre_classifier_model"
        )

        # ---------------------------------
        # Reports
        # ---------------------------------

        print("\nTraining Complete")
        print("--------------------------")

        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report")
        print("--------------------------")

        print(
            classification_report(
                y_test,
                predictions
            )
        )

        print("\nConfusion Matrix")
        print("--------------------------")

        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        print("\nSaved model to:")
        print(MODEL_OUTPUT)


if __name__ == "__main__":

    main()
