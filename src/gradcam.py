from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn


class GradCAM:
    def disable_inplace_relu(model):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        return model

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()

    def generate(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        denominator = cam.max() + 1e-8
        return cam / denominator


def make_heatmap_overlay(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.42) -> Image.Image:
    image = original_image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    image_np = np.asarray(image).astype(np.float32)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = (1 - alpha) * image_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def generate_gradcam_overlay(
    model = disable_inplace_relu(model),
    model: nn.Module,
    target_layer: nn.Module,
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    class_idx: int,
) -> Image.Image:
    gradcam = GradCAM(model, target_layer)
    try:
        cam = gradcam.generate(image_tensor, class_idx=class_idx)
    finally:
        gradcam.remove_hooks()
    return make_heatmap_overlay(original_image, cam)

