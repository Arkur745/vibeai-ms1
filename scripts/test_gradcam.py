import cv2
import torch

from app.ml.inference import (
    genre_model,
    create_spectrogram_image,
    device
)

from app.ml.gradcam_utils import (
    generate_gradcam
)


audio_path = "data/test_song.mp3"

image_path = create_spectrogram_image(
    audio_path
)

heatmap = generate_gradcam(

    genre_model,

    image_path,

    device
)

cv2.imwrite(

    "artifacts/gradcam_result.jpg",

    cv2.cvtColor(

        heatmap,

        cv2.COLOR_RGB2BGR
    )
)

print("\nGradCAM saved:")
print("artifacts/gradcam_result.jpg")
