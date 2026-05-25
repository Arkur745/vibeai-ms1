import pandas as pd

from tqdm import tqdm
from pathlib import Path

from app.ml.inference import (
    extract_features
)


METADATA_PATH = Path(
    "data/processed/gtzan_metadata.csv"
)

OUTPUT_PATH = Path(
    "data/features/gtzan_features_v1.csv"
)


def main():

    print("\nLoading GTZAN metadata...")

    df = pd.read_csv(
        METADATA_PATH
    )

    records = []

    failed = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        try:

            features = extract_features(
                row["audio_path"]
            )

            features["genre"] = row["genre"]

            records.append(features)

        except Exception as e:

            failed += 1

            print(f"\nFailed: {row['audio_path']}")
            print(e)

    # ---------------------------------
    # Build Dataset
    # ---------------------------------

    feature_df = pd.DataFrame(
        records
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------
    # Summary
    # ---------------------------------

    print("\nFeature Dataset Summary")
    print("----------------------------")

    print(feature_df.head())

    print(f"\nProcessed Samples: {len(feature_df)}")

    print(f"Failed Samples: {failed}")

    print(f"\nSaved feature dataset to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":

    main()
