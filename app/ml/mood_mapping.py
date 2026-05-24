import yaml

from pathlib import Path

MOOD_MAPPING_VERSION = "v1"

CONFIG_PATH = Path(
    "configs/mood_mapping_v1.yaml"
)


with open(CONFIG_PATH, "r") as file:

    config = yaml.safe_load(file)


def classify_mood(
    valence,
    arousal
):

    moods = config["moods"]

    for _, mood_data in moods.items():

        valence_min = mood_data.get(
            "valence_min",
            -999
        )

        valence_max = mood_data.get(
            "valence_max",
            999
        )

        arousal_min = mood_data.get(
            "arousal_min",
            -999
        )

        arousal_max = mood_data.get(
            "arousal_max",
            999
        )

        if (
            valence >= valence_min
            and valence < valence_max
            and arousal >= arousal_min
            and arousal < arousal_max
        ):

            return {
                "mood": mood_data["mood"],
                "vibe": mood_data["vibe"]
            }

    return {
        "mood": "Unknown",
        "vibe": "Undefined"
    }


def get_mood_mapping_version():

    return MOOD_MAPPING_VERSION
