import pandas as pd

from pathlib import Path
from tqdm import tqdm


DATASET_PATH = Path(
    "data/raw/gtzan"
)

OUTPUT_PATH = Path(
    "data/processed/gtzan_metadata.csv"
)


records = []


print("\nScanning GTZAN dataset...")


for genre_dir in DATASET_PATH.iterdir():

    if not genre_dir.is_dir():
        continue

    genre = genre_dir.name

    for audio_file in tqdm(
        genre_dir.glob("*.wav"),
        desc=genre
    ):

        records.append({

            "audio_path": str(audio_file),

            "genre": genre
        })


df = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nDataset Summary")
print("------------------------")

print(df.head())

print(f"\nTotal Samples: {len(df)}")

print(f"\nSaved metadata to:")
print(OUTPUT_PATH)
