import joblib
import librosa
import numpy as np
import pandas as pd
import shap

from app.ml.explanation_engine import (
    generate_explanation
)
from app.ml.mood_mapping import classify_mood

from app.core.config import (
    VALENCE_MODEL_PATH,
    AROUSAL_MODEL_PATH,
    GENRE_MODEL_PATH
)

# ---------------------------------
# Load Models
# ---------------------------------

print("Loading trained models...")

valence_model = joblib.load(
    VALENCE_MODEL_PATH
)

arousal_model = joblib.load(
    AROUSAL_MODEL_PATH
)

genre_bundle = joblib.load(
    GENRE_MODEL_PATH
)
print(AROUSAL_MODEL_PATH)
genre_model = genre_bundle["model"]

genre_encoder = genre_bundle["label_encoder"]

# ---------------------------------
# SHAP Explainer
# ---------------------------------

arousal_explainer = shap.TreeExplainer(
    arousal_model
)

# ---------------------------------
# Feature Extraction
# ---------------------------------

def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
       
    )

    # ---------------------------------
    # MFCC
    # ---------------------------------

    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfccs,
        axis=1
    )

    # ---------------------------------
    # Chroma
    # ---------------------------------

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    chroma_mean = np.mean(
        chroma,
        axis=1
    )

    # ---------------------------------
    # Spectral Centroid
    # ---------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    centroid_mean = np.mean(
        spectral_centroid
    )

    # ---------------------------------
    # Zero Crossing Rate
    # ---------------------------------

    zcr = librosa.feature.zero_crossing_rate(y)

    zcr_mean = np.mean(zcr)

    # ---------------------------------
    # RMS Energy
    # ---------------------------------

    rms = librosa.feature.rms(y=y)

    rms_mean = np.mean(rms)

    # ---------------------------------
    # Spectral Rolloff
    # ---------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    rolloff_mean = np.mean(rolloff)

    # ---------------------------------
    # Onset Strength
    # ---------------------------------

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr
    )

    onset_mean = np.mean(onset_env)
    # ---------------------------------
    # Tempo
    # ---------------------------------

    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    # ---------------------------------
    # Normalize Double Tempo Issue
    # ---------------------------------

    if tempo > 180:

        tempo = tempo / 2

    # ---------------------------------
    # Harmonic / Percussive Separation
    # ---------------------------------

    harmonic, percussive = librosa.effects.hpss(y)

    harmonic_energy = np.mean(
        np.abs(harmonic)
    )

    percussive_energy = np.mean(
        np.abs(percussive)
    )

    # ---------------------------------
    # Tonnetz
    # ---------------------------------

    tonnetz = librosa.feature.tonnetz(
        y=harmonic,
        sr=sr
    )

    tonnetz_mean = np.mean(
        tonnetz,
        axis=1
    )

    feature_dict = {}

    # ---------------------------------
    # MFCC Features
    # ---------------------------------

    for i, value in enumerate(mfcc_mean):

        feature_dict[f"mfcc_{i+1}"] = value

    # ---------------------------------
    # Chroma Features
    # ---------------------------------

    for i, value in enumerate(chroma_mean):

        feature_dict[f"chroma_{i+1}"] = value

    # ---------------------------------
    # Basic DSP Features
    # ---------------------------------

    feature_dict["spectral_centroid"] = centroid_mean

    feature_dict["zero_crossing_rate"] = zcr_mean

    feature_dict["rms_energy"] = rms_mean

    feature_dict["spectral_rolloff"] = rolloff_mean

    feature_dict["onset_strength"] = onset_mean

    feature_dict["harmonic_energy"] = harmonic_energy

    feature_dict["percussive_energy"] = percussive_energy
    
    feature_dict["tempo"] = float(tempo)

    # ---------------------------------
    # Tonnetz Features
    # ---------------------------------

    for i, value in enumerate(tonnetz_mean):

        feature_dict[f"tonnetz_{i+1}"] = value

    print("\nFeature Count:")
    print(len(feature_dict))

    return feature_dict


# ---------------------------------
# Prediction Pipeline
# ---------------------------------

def predict_emotion(audio_path):

    features = extract_features(
        audio_path
    )

    # ---------------------------------
    # Preserve Tempo For API Response
    # ---------------------------------

    tempo_value = features["tempo"]

    # ---------------------------------
    # Emotion Model Features
    # (without tempo)
    # ---------------------------------

    emotion_features = features.copy()

    del emotion_features["tempo"]

    X_emotion = pd.DataFrame(
        [emotion_features]
    )

    # ---------------------------------
    # Genre Model Features
    # (with tempo)
    # ---------------------------------

    X_genre = pd.DataFrame(
        [features]
    )

    # ---------------------------------
    # Debug
    # ---------------------------------

    print("\nEmotion Features:")
    print(X_emotion.columns.tolist())

    print("\nEmotion Feature Count:")
    print(len(X_emotion.columns))

    print("\nGenre Features:")
    print(X_genre.columns.tolist())

    print("\nGenre Feature Count:")
    print(len(X_genre.columns))

    # ---------------------------------
    # SHAP Explanation
    # ---------------------------------

    shap_values = arousal_explainer.shap_values(
        X_emotion
    )

    # ---------------------------------
    # Emotion Predictions
    # ---------------------------------

    valence = valence_model.predict(
        X_emotion
    )[0]

    arousal = arousal_model.predict(
        X_emotion
    )[0]

    # ---------------------------------
    # Genre Prediction
    # ---------------------------------

    genre_prediction = genre_model.predict(
        X_genre
    )[0]

    genre = genre_encoder.inverse_transform(
        [genre_prediction]
    )[0]

    # ---------------------------------
    # Mood Mapping
    # ---------------------------------

    mood_data = classify_mood(

        float(valence),

        float(arousal)
    )

    # ---------------------------------
    # SHAP Contribution Analysis
    # ---------------------------------

    contributions = pd.DataFrame({

        "feature": X_emotion.columns,

        "shap_value": shap_values[0]
    })

    contributions["abs_value"] = (

        contributions["shap_value"].abs()
    )

    contributions = contributions.sort_values(

        by="abs_value",

        ascending=False
    )

    # ---------------------------------
    # Human Readable Explanations
    # ---------------------------------

    explanations = generate_explanation(
        contributions
    )

    # ---------------------------------
    # Final Response
    # ---------------------------------

    result = {

        "genre": genre,

        "valence": round(float(valence), 3),

        "arousal": round(float(arousal), 3),

        "mood": mood_data["mood"],

        "vibe": mood_data["vibe"],

        "explanations": explanations,

        "tempo": round(float(tempo_value), 2)
    }

    return result

# ---------------------------------
# CLI Testing
# ---------------------------------

if __name__ == "__main__":

    audio_file = "data/test_song.mp3"

    prediction = predict_emotion(
        audio_file
    )

    print("\nEmotion Prediction")
    print("----------------------")

    print(prediction)
