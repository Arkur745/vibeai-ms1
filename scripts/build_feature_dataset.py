import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import mlflow


METADATA_PATH = Path("data/processed/deam_metadata.csv")

OUTPUT_PATH = Path("data/features/deam_features.csv")


def extract_features(audio_path):

    y, sr = librosa.load(audio_path, duration=30)

    # -----------------------------
    # MFCC
    # -----------------------------
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfccs, axis=1)

    # -----------------------------
    # Chroma
    # -----------------------------
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    chroma_mean = np.mean(chroma, axis=1)

    # -----------------------------
    # Spectral Centroid
    # -----------------------------
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    centroid_mean = np.mean(spectral_centroid)

    # -----------------------------
    # Zero Crossing Rate
    # -----------------------------
    zcr = librosa.feature.zero_crossing_rate(y)

    zcr_mean = np.mean(zcr)

    feature_dict = {}

    # MFCC features
    for i, value in enumerate(mfcc_mean):
        feature_dict[f"mfcc_{i+1}"] = value

    # Chroma features
    for i, value in enumerate(chroma_mean):
        feature_dict[f"chroma_{i+1}"] = value

    feature_dict["spectral_centroid"] = centroid_mean
    feature_dict["zero_crossing_rate"] = zcr_mean

    return feature_dict


def main():

    metadata_df = pd.read_csv(METADATA_PATH)

    records = []

    with mlflow.start_run(run_name="feature_extraction_pipeline"):

        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):

            try:

                features = extract_features(row["audio_path"])

                features["song_id"] = row["song_id"]
                features["valence"] = row["valence"]
                features["arousal"] = row["arousal"]

                records.append(features)

            except Exception as e:
                print(f"\nFailed: {row['audio_path']}")
                print(e)

        features_df = pd.DataFrame(records)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        features_df.to_csv(OUTPUT_PATH, index=False)

        mlflow.log_metric(
            "processed_samples",
            len(features_df)
        )

        mlflow.log_param(
            "feature_count",
            len(features_df.columns)
        )

        print("\nFeature Dataset Summary")
        print("----------------------------")

        print(features_df.head())

        print(f"\nSaved feature dataset to:")
        print(OUTPUT_PATH)


if __name__ == "__main__":
    main()