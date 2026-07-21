"""Grad-CAM (+ CBAM attention, where applicable) for the /explain endpoint.

Ported from 07_diagnostics.ipynb — same retain_grad()-based implementation
(register_full_backward_hook conflicts with the in-place ReLU DenseNet121
applies right after its feature extractor).
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import DEVICE, get_model, preprocess

IMG_SIZE = 224


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.fwd = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        output.retain_grad()
        self.activations = output

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        output[0, class_idx].backward()

        gradients  = self.activations.grad[0]
        activation = self.activations.detach()[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.einsum("c,chw->hw", weights, activation)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx

    def remove(self):
        self.fwd.remove()


TARGET_LAYER = {
    "custom_cnn":           lambda m: m.conv6,
    "custom_cnn_tuned":     lambda m: m.conv6,
    "convnext_tiny":        lambda m: m.features,
    "densenet121":          lambda m: m.features,
    "efficientnet_b0_cbam": lambda m: m.cbam,
    "mobilenetv3_small":    lambda m: m.features,
}


def overlay_cam(pil_image, cam, alpha=0.45):
    img  = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    heat = plt.get_cmap("inferno")(np.array(heat) / 255.0)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def _array_to_data_uri(arr):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def explain(model_key, pil_image):
    """Returns {"gradcam": data-uri, "cbam_attention": data-uri | None}."""
    model  = get_model(model_key)
    tensor = preprocess(pil_image).to(DEVICE)

    gradcam = GradCAM(model, TARGET_LAYER[model_key](model))
    cam, _  = gradcam(tensor)
    gradcam.remove()

    result = {
        "gradcam": _array_to_data_uri(overlay_cam(pil_image, cam)),
        "cbam_attention": None,
    }

    if model_key == "efficientnet_b0_cbam":
        captured = {}

        def _hook(module, inputs, output):
            captured["attn"] = output.detach()

        handle = model.cbam.spatial_attention.register_forward_hook(_hook)
        with torch.no_grad():
            model(tensor)
        handle.remove()

        attn = captured["attn"][0, 0].cpu().numpy()
        result["cbam_attention"] = _array_to_data_uri(overlay_cam(pil_image, attn))

    return result
