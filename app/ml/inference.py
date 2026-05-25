import joblib
import librosa
import numpy as np
import pandas as pd
from app.ml.mood_mapping import classify_mood

from app.core.config import (
    VALENCE_MODEL_PATH,
    AROUSAL_MODEL_PATH
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


# ---------------------------------
# Feature Extraction
# ---------------------------------

def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        duration=30
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

    X = pd.DataFrame([features])

    valence = valence_model.predict(X)[0]

    arousal = arousal_model.predict(X)[0]

    mood_data = classify_mood(
        float(valence),
        float(arousal)
    )


    result = {
    
        "valence": round(float(valence), 3),
    
        "arousal": round(float(arousal), 3),
    
        "mood": mood_data["mood"],
    
        "vibe": mood_data["vibe"]
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
