import joblib
import pandas as pd

from pathlib import Path


# ---------------------------------
# Paths
# ---------------------------------

MODEL_PATH = Path(
    "models/random_forest_arousal_v2.pkl"
)

FEATURE_DATASET = Path(
    "data/features/deam_features_v2.csv"
)


# ---------------------------------
# Load Model
# ---------------------------------

print("\nLoading trained model...")

model = joblib.load(
    MODEL_PATH
)


# ---------------------------------
# Load Dataset
# ---------------------------------

print("Loading feature dataset...")

df = pd.read_csv(
    FEATURE_DATASET
)

X = df.drop(
    columns=[
        "song_id",
        "valence",
        "arousal"
    ]
)


# ---------------------------------
# Feature Importance
# ---------------------------------

importance_df = pd.DataFrame({

    "feature": X.columns,

    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# ---------------------------------
# Display Results
# ---------------------------------

print("\nTop 20 Most Important Features")
print("----------------------------------")

print(
    importance_df.head(20)
)
