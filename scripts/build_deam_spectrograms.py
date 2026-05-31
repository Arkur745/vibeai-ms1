from pathlib import Path

import numpy as np
import pandas as pd

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm


# ---------------------------------
# Paths
# ---------------------------------

CSV_PATH = (
    "data/features/deam_features_v2.csv"
)

AUDIO_DIR = Path(
    "data/raw/deam/audio"
)

OUTPUT_DIR = Path(
    "data/spectrograms/deam"
)

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True
)

OUTPUT_CSV = (
    "data/processed/deam_spectrograms.csv"
)


# ---------------------------------
# Spectrogram Generator
# ---------------------------------

def generate_spectrogram(

    audio_path,

    output_path
):

    try:

        y, sr = librosa.load(

            audio_path,

            sr=22050
        )

        mel = librosa.feature.melspectrogram(

            y=y,

            sr=sr,

            n_mels=128
        )

        mel_db = librosa.power_to_db(

            mel,

            ref=np.max
        )

        plt.figure(

            figsize=(4, 4)
        )

        librosa.display.specshow(

            mel_db,

            sr=sr,

            x_axis=None,

            y_axis=None
        )

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(

            output_path,

            bbox_inches="tight",

            pad_inches=0
        )

        plt.close()

        return True

    except Exception as e:

        print(f"\nFailed: {audio_path}")
        print(e)

        return False


# ---------------------------------
# Main
# ---------------------------------

def main():

    df = pd.read_csv(CSV_PATH)

    rows = []

    for idx, row in tqdm(

        df.iterrows(),

        total=len(df)
    ):

        # ---------------------------------
        # Build Audio Path From song_id
        # ---------------------------------

        song_id = int(
            float(row["song_id"])
        )

        # ---------------------------------
        # Try Different Filename Formats
        # ---------------------------------

        possible_paths = [

            AUDIO_DIR / f"{song_id}.mp3",

            AUDIO_DIR / f"{song_id}.wav",

            AUDIO_DIR / f"{song_id:04d}.mp3",

            AUDIO_DIR / f"{song_id:04d}.wav"
        ]

        audio_path = None

        for path in possible_paths:

            if path.exists():

                audio_path = path

                break

        # ---------------------------------
        # Skip Missing Files
        # ---------------------------------

        if audio_path is None:

            continue

        valence = row["valence"]

        arousal = row["arousal"]

        filename = (
            audio_path.stem + ".png"
        )

        output_path = (
            OUTPUT_DIR / filename
        )

        success = generate_spectrogram(

            audio_path,

            output_path
        )

        if success:

            rows.append({

                "image_path": str(output_path),

                "valence": valence,

                "arousal": arousal
            })

    # ---------------------------------
    # Save Metadata CSV
    # ---------------------------------

    final_df = pd.DataFrame(rows)

    Path("data/processed").mkdir(
        exist_ok=True
    )

    final_df.to_csv(

        OUTPUT_CSV,

        index=False
    )

    print("\nSaved:")
    print(OUTPUT_CSV)

    print("\nTotal Spectrograms:")
    print(len(final_df))


if __name__ == "__main__":

    main()
