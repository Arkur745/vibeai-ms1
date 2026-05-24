import librosa
import numpy as np
import sys
from pathlib import Path


def extract_audio_features(audio_path):
    print(f"\nLoading: {audio_path}")

    # Load first 30 seconds
    y, sr = librosa.load(audio_path, duration=30)

    print(f"Sample Rate: {sr}")
    print(f"Audio Length: {len(y)} samples")

    # -----------------------------------
    # MFCCs
    # -----------------------------------
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfccs, axis=1)

    print("\nMFCC Means:")
    print(mfcc_mean)

    # -----------------------------------
    # Chroma Features
    # -----------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    chroma_mean = np.mean(chroma, axis=1)

    print("\nChroma Means:")
    print(chroma_mean)

    # -----------------------------------
    # Spectral Centroid
    # -----------------------------------
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    centroid_mean = np.mean(spectral_centroid)

    print(f"\nSpectral Centroid Mean: {centroid_mean:.2f}")

    # -----------------------------------
    # Zero Crossing Rate
    # -----------------------------------
    zcr = librosa.feature.zero_crossing_rate(y)

    zcr_mean = np.mean(zcr)

    print(f"Zero Crossing Rate Mean: {zcr_mean:.5f}")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts/extract_features.py path/to/song.mp3")
        sys.exit(1)

    audio_file = Path(sys.argv[1])

    if not audio_file.exists():
        print("File does not exist.")
        sys.exit(1)

    extract_audio_features(audio_file)