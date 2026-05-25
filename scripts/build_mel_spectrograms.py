import matplotlib.pyplot as plt
import librosa.display
import librosa
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from tqdm import tqdm

import pandas as pd


DATASET_PATH = Path(
    "data/raw/gtzan"
)

OUTPUT_DIR = Path(
    "data/spectrograms"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def generate_mel_spectrogram(

    audio_path,

    output_path
):

    y, sr = librosa.load(

        audio_path,

        duration=30
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
        figsize=(3, 3)
    )

    librosa.display.specshow(

        mel_db,

        sr=sr,

        x_axis="time",

        y_axis="mel"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(

        output_path,

        bbox_inches="tight",

        pad_inches=0
    )

    plt.close()


def main():

    metadata = []

    genres = sorted([
        d.name for d in DATASET_PATH.iterdir()
        if d.is_dir()
    ])

    for genre in genres:

        genre_dir = DATASET_PATH / genre

        output_genre_dir = OUTPUT_DIR / genre

        output_genre_dir.mkdir(
            exist_ok=True
        )

        files = list(
            genre_dir.glob("*.wav")
        )

        for file_path in tqdm(
            files,
            desc=genre
        ):

            output_path = (
                output_genre_dir /
                f"{file_path.stem}.png"
            )

            try:

                generate_mel_spectrogram(
                
                    file_path,
            
                    output_path
                )
            
                metadata.append({
                
                    "image_path": str(output_path),
            
                    "genre": genre
                })
            
            
            except Exception as e:
            
                print(f"\nFailed: {file_path}")
            
                print(e)
            
                continue

            metadata.append({

                "image_path": str(output_path),

                "genre": genre
            })

    df = pd.DataFrame(metadata)

    df.to_csv(

        "data/processed/gtzan_spectrograms.csv",

        index=False
    )

    print("\nSpectrogram generation complete.")


if __name__ == "__main__":

    import numpy as np

    main()
