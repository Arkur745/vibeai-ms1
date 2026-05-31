import cv2
import numpy as np
import torch

from PIL import Image

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)

from torchvision import transforms


# ---------------------------------
# Image Transform
# ---------------------------------

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------
# Generate GradCAM
# ---------------------------------

def generate_gradcam(

    model,

    image_path,

    device
):

    # ---------------------------------
    # Load Image
    # ---------------------------------

    rgb_image = Image.open(

        image_path

    ).convert("RGB")

    rgb_image = rgb_image.resize(
        (224, 224)
    )

    rgb_np = np.array(

        rgb_image

    ).astype(np.float32) / 255.0

    input_tensor = transform(
        rgb_image
    ).unsqueeze(0).to(device)

    # ---------------------------------
    # Target Layer
    # ---------------------------------

    target_layers = [

        model.model.layer4[-1]
    ]

    # ---------------------------------
    # Prediction
    # ---------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        predicted_class = torch.argmax(

            output,

            dim=1

        ).item()

    # ---------------------------------
    # GradCAM
    # ---------------------------------

    cam = GradCAM(

        model=model,

        target_layers=target_layers
    )

    targets = [

        ClassifierOutputTarget(
            predicted_class
        )
    ]

    grayscale_cam = cam(

        input_tensor=input_tensor,

        targets=targets
    )[0]

    # ---------------------------------
    # Overlay Heatmap
    # ---------------------------------

    visualization = show_cam_on_image(

        rgb_np,

        grayscale_cam,

        use_rgb=True
    )

    return visualization
