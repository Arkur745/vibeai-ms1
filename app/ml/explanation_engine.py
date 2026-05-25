FEATURE_EXPLANATIONS = {

    "percussive_energy":
        "strong percussion and drum activity",

    "onset_strength":
        "sharp rhythmic attacks",

    "spectral_rolloff":
        "bright high-frequency sound characteristics",

    "spectral_centroid":
        "bright and energetic tonal texture",

    "harmonic_energy":
        "rich melodic and harmonic content",

    "rms_energy":
        "high overall loudness and intensity",

    "zero_crossing_rate":
        "noisy or aggressive sound textures",

    "mfcc_1":
        "strong low-level timbral texture",

    "mfcc_2":
        "dynamic tonal coloration",

    "mfcc_3":
        "emotionally expressive tonal patterns"
}


def generate_explanation(contribution_df):

    explanations = []

    top_features = contribution_df.head(5)

    for _, row in top_features.iterrows():

        feature = row["feature"]

        shap_value = row["shap_value"]

        if feature not in FEATURE_EXPLANATIONS:
            continue

        explanation_text = FEATURE_EXPLANATIONS[
            feature
        ]

        if shap_value > 0:

            explanations.append(
                f"Higher {explanation_text} increased the emotional intensity."
            )

        else:

            explanations.append(
                f"Lower {explanation_text} reduced the emotional intensity."
            )

    return explanations
