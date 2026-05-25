import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import mlflow


METADATA_PATH = Path("data/processed/deam_metadata.csv")

OUTPUT_PATH = Path(
    "data/features/deam_features_v2.csv"
)


def extract_features(audio_path):

    # Load first 30 seconds
    y, sr = librosa.load(audio_path, duration=30)

    # Validate audio length
    if len(y) < sr * 5:
        raise ValueError("Audio too short")

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

    # -----------------------------
    # Store MFCC Features
    # -----------------------------
    for i, value in enumerate(mfcc_mean):
        feature_dict[f"mfcc_{i+1}"] = value

    # -----------------------------
    # Store Chroma Features
    # -----------------------------
    for i, value in enumerate(chroma_mean):
        feature_dict[f"chroma_{i+1}"] = value

    # -----------------------------
    # Store Additional Features
    # -----------------------------
    feature_dict["spectral_centroid"] = centroid_mean
    feature_dict["zero_crossing_rate"] = zcr_mean


    feature_dict["rms_energy"] = rms_mean

    feature_dict["spectral_rolloff"] = rolloff_mean

    feature_dict["onset_strength"] = onset_mean

    feature_dict["harmonic_energy"] = harmonic_energy

    feature_dict["percussive_energy"] = percussive_energy


    # Tonnetz Features
    for i, value in enumerate(tonnetz_mean):

        feature_dict[f"tonnetz_{i+1}"] = value
    return feature_dict


def main():

    metadata_df = pd.read_csv(METADATA_PATH)

    records = []

    with mlflow.start_run(run_name="feature_extraction_pipeline"):

        failed_count = 0
        mlflow.log_param(
            "feature_version",
            "v2"
        )

        for _, row in tqdm(
            metadata_df.iterrows(),
            total=len(metadata_df)
        ):

            try:

                features = extract_features(
                    row["audio_path"]
                )

                # Add labels
                features["song_id"] = row["song_id"]
                features["valence"] = row["valence"]
                features["arousal"] = row["arousal"]

                records.append(features)

            except Exception as e:

                failed_count += 1

                print(f"\nFailed: {row['audio_path']}")
                print(e)

        # --------------------------------
        # Build DataFrame
        # --------------------------------
        features_df = pd.DataFrame(records)

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        features_df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        # --------------------------------
        # MLflow Logging
        # --------------------------------
        mlflow.log_metric(
            "processed_samples",
            len(features_df)
        )

        mlflow.log_metric(
            "failed_samples",
            failed_count
        )

        mlflow.log_param(
            "feature_count",
            len(features_df.columns)
        )

        # --------------------------------
        # Console Output
        # --------------------------------
        print("\nFeature Dataset Summary")
        print("----------------------------")

        print(features_df.head())

        print(f"\nProcessed Samples: {len(features_df)}")
        print(f"Failed Samples: {failed_count}")

        print(f"\nSaved feature dataset to:")
        print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
