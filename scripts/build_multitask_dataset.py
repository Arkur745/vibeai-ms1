from pathlib import Path

import pandas as pd


# ---------------------------------
# Load GTZAN
# ---------------------------------

gtzan_df = pd.read_csv(
    "data/processed/gtzan_spectrograms.csv"
)

gtzan_df["valence"] = -1
gtzan_df["arousal"] = -1

# ---------------------------------
# Load DEAM
# ---------------------------------

deam_df = pd.read_csv(
    "data/processed/deam_spectrograms.csv"
)

deam_df["genre"] = "unknown"

# ---------------------------------
# Standardize Columns
# ---------------------------------

gtzan_df = gtzan_df[[

    "image_path",

    "genre",

    "valence",

    "arousal"
]]

deam_df = deam_df[[

    "image_path",

    "genre",

    "valence",

    "arousal"
]]

# ---------------------------------
# Combine
# ---------------------------------

combined_df = pd.concat(

    [gtzan_df, deam_df],

    ignore_index=True
)

# ---------------------------------
# Save
# ---------------------------------

Path("data/processed").mkdir(
    exist_ok=True
)

combined_df.to_csv(

    "data/processed/multitask_dataset.csv",

    index=False
)

print("\nSaved multitask dataset.")
print(combined_df.head())

print("\nTotal Samples:")
print(len(combined_df))
