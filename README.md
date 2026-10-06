# ComfyUI-ZipDepth

A simple implementation of [ZipDepth](https://github.com/fabiotosi92/ZipDepth) monocular depth estimation for ComfyUI.
Lightweight (6.1M params), fast zero-shot relative depth — runs on GPU or CPU.

![preview](main.png?raw=true "ComfyUI-ZipDepth")

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/ComfyUI-ZipDepth.git
```

Restart ComfyUI. The `zipdepth_base.pth` checkpoint (~27 MB) auto-downloads from the
[official repo](https://github.com/fabiotosi92/ZipDepth) on first use if not already bundled.

## Usage

Node: **ZipDepth Estimate** (category `ZipDepth`)

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | Input image(s) |
| `colormap` | Combo | Colormap for `depth_colormap` (Spectral, Turbo, Inferno, Viridis, Magma) |
| `invert` | BOOLEAN | Default: near = white, far = black. Enable to flip |

| Output | Type | Description |
|---|---|---|
| `depth` | IMAGE | Grayscale depth, min-max normalized per image (near = 1.0) |
| `depth_colormap` | IMAGE | Colormapped depth visualization |

Images are resized to a shorter side of ~384px (multiple of 32) for the model,
and the depth map is resized back to the input resolution.

## Credits

- ZipDepth by Fabio Tosi et al. — original code and models:
  https://github.com/fabiotosi92/ZipDepth
- Architecture from Tongji University researchers.

## License

MIT
