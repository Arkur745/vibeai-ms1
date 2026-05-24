import pandas as pd
import joblib
from pathlib import Path


MODEL_PATH = Path("models/valence_model.pkl")

FEATURE_DATASET = Path(
    "data/features/deam_features.csv"
)


def main():

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Loading feature dataset...")

    df = pd.read_csv(FEATURE_DATASET)

    X = df.drop(
        columns=[
            "song_id",
            "valence",
            "arousal"
        ]
    )

    feature_names = X.columns

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print("\nTop 15 Most Important Features")
    print("----------------------------------")

    print(
        importance_df.head(15)
    )


if __name__ == "__main__":
    main()