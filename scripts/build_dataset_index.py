import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/raw/deam")

LABEL_1 = DATA_DIR / "labels/static_annotations_averaged_songs_1_2000.csv"
LABEL_2 = DATA_DIR / "labels/static_annotations_averaged_songs_2000_2058.csv"

AUDIO_DIR = DATA_DIR / "audio"

OUTPUT_PATH = Path("data/processed/deam_metadata.csv")


def load_labels():
    print("\nLoading label CSV files...")

    df1 = pd.read_csv(LABEL_1)
    df2 = pd.read_csv(LABEL_2)


    # Clean column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df = pd.concat([df1, df2], ignore_index=True)

    print(f"Total label rows: {len(df)}")

    return df


def build_metadata(df):
    records = []

    print("\nMatching audio files...")

    for _, row in df.iterrows():

        song_id = int(row["song_id"])
    
        audio_path = AUDIO_DIR / f"{song_id}.mp3"
    
        if not audio_path.exists():
            continue
        
        record = {
            "song_id": song_id,
            "audio_path": str(audio_path),
            "valence": row["valence_mean"],
            "arousal": row["arousal_mean"]
        }
    
        records.append(record)

    metadata_df = pd.DataFrame(records)

    return metadata_df


def main():

    df = load_labels()

    metadata_df = build_metadata(df)

    print("\nDataset Summary")
    print("------------------------")

    print(metadata_df.head())

    print(f"\nFinal matched samples: {len(metadata_df)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved metadata to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()