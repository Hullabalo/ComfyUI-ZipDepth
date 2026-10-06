"""ZipDepth monocular depth estimation for ComfyUI. Original work: https://github.com/fabiotosi92/ZipDepth"""

import os
import urllib.request

import torch
import torch.nn.functional as F

from .zipdepth_model import ZipDepth
from .colormaps import COLORMAPS, get_lut

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(NODE_DIR, "checkpoints", "zipdepth_base.pth")
CKPT_URL = "https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth"

_model = None  # cached loaded model


def _get_model():
    global _model
    if _model is None:
        if not os.path.isfile(CKPT_PATH):
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            print(f"[ZipDepth] downloading {CKPT_URL}")
            urllib.request.urlretrieve(CKPT_URL, CKPT_PATH)
        model = ZipDepth()
        model.load_state_dict(
            torch.load(CKPT_PATH, map_location="cpu", weights_only=True), strict=True)
        model.fuse_for_inference().requires_grad_(False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = model.to(device)
    return _model


class ZipDepthEstimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormap": (COLORMAPS, {"default": "Spectral"}),
                "invert": ("BOOLEAN", {"default": False,
                                       "tooltip": "Default: near = white. Enable for near = black."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("depth", "depth_colormap")
    FUNCTION = "estimate"
    CATEGORY = "ZipDepth"
    DESCRIPTION = "Relative monocular depth. 'depth' is min-max normalized per image (near = 1.0)."

    def estimate(self, image, colormap, invert):
        model = _get_model()
        device = next(model.parameters()).device

        # ComfyUI IMAGE: [B, H, W, 3] float 0-1 RGB -> [B, 3, H/2*32-ish, ...]
        x = image[..., :3].permute(0, 3, 1, 2)
        B, _, H, W = x.shape
        scale = 384 / min(H, W)
        new_h = max(32, int(round(H * scale / 32)) * 32)
        new_w = max(32, int(round(W * scale / 32)) * 32)

        with torch.no_grad():
            xb = F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                               align_corners=False).clamp(0, 1).to(device)
            depth = model(xb).float()          # [B, 1, h, w]
            depth = F.interpolate(depth, size=(H, W), mode="bilinear",
                                  align_corners=True)[:, 0].cpu()

        raw = depth.reshape(B, -1)
        dmin = raw.min(dim=1).values.view(B, 1, 1)
        dmax = raw.max(dim=1).values.view(B, 1, 1)
        norm = ((depth - dmin) / (dmax - dmin + 1e-8)).clamp(0, 1)
        if invert:
            norm = 1 - norm

        depth_img = norm.unsqueeze(-1).repeat(1, 1, 1, 3).contiguous()

        lut = torch.from_numpy(get_lut(colormap)).float() / 255.0
        idx = (255 - norm.clamp(0, 1) * 255).round().long().clamp(0, 255)
        color_img = lut[idx].contiguous()

        return (depth_img, color_img)


NODE_CLASS_MAPPINGS = {"ZipDepthEstimate": ZipDepthEstimate}
NODE_DISPLAY_NAME_MAPPINGS = {"ZipDepthEstimate": "ZipDepth Estimate"}
