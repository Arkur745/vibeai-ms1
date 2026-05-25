import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ---------------------------------
# Paths
# ---------------------------------

MODEL_PATH = Path(
    "models/xgboost_arousal_v2.pkl"
)

DATASET_PATH = Path(
    "data/features/deam_features_v2.csv"
)


# ---------------------------------
# Load Model
# ---------------------------------

print("\nLoading model...")

model = joblib.load(
    MODEL_PATH
)


# ---------------------------------
# Load Dataset
# ---------------------------------

print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

X = df.drop(
    columns=[
        "song_id",
        "valence",
        "arousal"
    ]
)


# ---------------------------------
# SHAP Explainer
# ---------------------------------

print("\nBuilding SHAP explainer...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X
)


# ---------------------------------
# Summary Plot
# ---------------------------------

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X,
    show=True
)
