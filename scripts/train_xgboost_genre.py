import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder


# ---------------------------------
# Paths
# ---------------------------------

DATA_PATH = Path(
    "data/features/gtzan_features_v1.csv"
)

MODEL_OUTPUT = Path(
    "models/xgboost_genre_classifier_v1.pkl"
)


def main():

    print("\nLoading genre dataset...")

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

    # ---------------------------------
    # Encode Labels
    # ---------------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    print("\nGenres:")
    print(encoder.classes_)

    print("\nFeature Shape:")
    print(X.shape)

    # ---------------------------------
    # Train Test Split
    # ---------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y_encoded,

        test_size=0.2,

        random_state=42,

        stratify=y_encoded
    )

    print(f"\nTraining samples: {len(X_train)}")

    print(f"Testing samples: {len(X_test)}")

    # ---------------------------------
    # MLflow Tracking
    # ---------------------------------

    with mlflow.start_run(
        run_name="xgboost_genre_classifier_v1"
    ):

        mlflow.log_param(
            "model_type",
            "XGBoostClassifier"
        )

        mlflow.log_param(
            "feature_version",
            "v1"
        )

        # ---------------------------------
        # Model
        # ---------------------------------

        model = XGBClassifier(

            n_estimators=400,

            learning_rate=0.05,

            max_depth=8,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42,

            n_jobs=-1
        )

        print("\nTraining XGBoost genre classifier...")

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
            {
                "model": model,
                "label_encoder": encoder
            },
            MODEL_OUTPUT
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="xgboost_genre_classifier"
        )

        # ---------------------------------
        # Reports
        # ---------------------------------

        print("\nTraining Complete")
        print("--------------------------")

        print(f"Accuracy: {accuracy:.4f}")

        decoded_predictions = encoder.inverse_transform(
            predictions
        )

        decoded_y_test = encoder.inverse_transform(
            y_test
        )

        print("\nClassification Report")
        print("--------------------------")

        print(

            classification_report(

                decoded_y_test,

                decoded_predictions
            )
        )

        print("\nConfusion Matrix")
        print("--------------------------")

        print(

            confusion_matrix(

                decoded_y_test,

                decoded_predictions
            )
        )

        print("\nSaved model to:")
        print(MODEL_OUTPUT)


if __name__ == "__main__":

    main()
