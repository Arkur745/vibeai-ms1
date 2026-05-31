import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from app.core.config import settings
from app.core.logger import logger

matplotlib.use("Agg")


def generate_gradcam(image_tensor: torch.Tensor, class_idx: int, task_id: str, original_image: Image.Image, model) -> str:
    activation = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activation["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    target_layer = model.model.layer4[1].conv2
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        image_tensor = image_tensor.requires_grad_(True)
        output = model(image_tensor)
        score = output[0, class_idx]
        model.zero_grad()
        score.backward(retain_graph=True)

        activations = activation["value"][0]
        grads = gradients["value"][0]
        weights = torch.mean(grads, dim=(1, 2))
        cam = torch.sum(weights[:, None, None] *
                        activations, dim=0).cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = cam / cam.max()

        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_resized = F.interpolate(
            cam_tensor,
            size=(original_image.height, original_image.width),
            mode="bilinear",
            align_corners=False,
        )
        cam_resized = cam_resized.squeeze().cpu().numpy()

        heatmap = plt.get_cmap("jet")(cam_resized)
        heatmap_image = Image.fromarray(
            (heatmap[:, :, :3] * 255).astype(np.uint8)).convert("RGB")
        overlay = Image.blend(original_image.convert(
            "RGB"), heatmap_image, alpha=0.4)

        gradcam_path = settings.gradcam_dir / f"gradcam_{task_id}.png"
        gradcam_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(gradcam_path)
        return str(gradcam_path)
    finally:
        forward_handle.remove()
        backward_handle.remove()
