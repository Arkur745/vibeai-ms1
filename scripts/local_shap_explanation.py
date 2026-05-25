import shap
import joblib
import pandas as pd

from pathlib import Path

from app.ml.inference import (
    extract_features
)
from app.ml.explanation_engine import (
    generate_explanation
)

import app.ml.inference
print(app.ml.inference.__file__)

# ---------------------------------
# Paths
# ---------------------------------

MODEL_PATH = Path(
    "models/xgboost_arousal_v2.pkl"
)

TEST_AUDIO = Path(
    "data/test_song.mp3"
)


# ---------------------------------
# Load Model
# ---------------------------------

print("\nLoading model...")

model = joblib.load(
    MODEL_PATH
)


# ---------------------------------
# Extract Features
# ---------------------------------

print("Extracting features...")

features = extract_features(
    TEST_AUDIO
)

X = pd.DataFrame(
    [features]
)


# ---------------------------------
# SHAP Explainer
# ---------------------------------

print("Generating SHAP explanation...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X
)


# ---------------------------------
# Prediction
# ---------------------------------

prediction = model.predict(
    X
)[0]


print("\nPrediction")
print("-------------------")

print(f"Arousal Score: {prediction:.3f}")


# ---------------------------------
# Feature Contributions
# ---------------------------------

contributions = pd.DataFrame({

    "feature": X.columns,

    "shap_value": shap_values[0]
})

contributions["abs_value"] = (
    contributions["shap_value"]
    .abs()
)

contributions = contributions.sort_values(
    by="abs_value",
    ascending=False
)


print("\nTop Feature Contributions")
print("-----------------------------")

print(
    contributions.head(10)
)
# ---------------------------------
# Human Readable Explanation
# ---------------------------------

explanations = generate_explanation(
    contributions
)

print("\nHuman Readable Explanation")
print("--------------------------------")

for explanation in explanations:

    print(f"- {explanation}")
