"""Inspect which image regions support each model prediction."""

import torch
import torch.nn.functional as F



def grad_cam(model, target_layer, image_tensor, target_class):
    """Return the normalized map for one image and one target class."""
    captured = {}

    def save_activation(module, inputs, output):
        captured["activation"] = output

    model.eval()
    handle = target_layer.register_forward_hook(save_activation)
    try:
        # Input gradients keep the graph available when model parameters are frozen.
        with torch.enable_grad():
            image_tensor = image_tensor.detach().requires_grad_(True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=image_tensor.device.type == "cuda"
            ):
                logits = model(image_tensor)
            gradient = torch.autograd.grad(logits[0, target_class], captured["activation"])[0]
            weights = gradient.float().mean(dim=(2, 3), keepdim=True)
            heatmap = (weights * captured["activation"].float()).sum(dim=1, keepdim=True).relu()
            heatmap = F.interpolate(
                heatmap, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
        heatmap = heatmap.detach().cpu().numpy()
        return heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    finally:
        handle.remove()
