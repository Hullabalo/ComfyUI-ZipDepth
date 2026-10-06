"""ZipDepth 'base' architecture — only what's needed to load checkpoints/zipdepth_base.pth."""

import torch
import torch.nn as nn
import torch.nn.functional as F

DIMS = [48, 96, 192, 384]
DEPTHS = [2, 2, 6, 2]
DEC_CH = 96
HALF_DEC_CH = 32


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class QARepBlock(nn.Module):
    """RepVGG block: 3x3 + 1x1 + identity -> fused 3x3 at inference."""
    def __init__(self, in_ch, out_ch, stride=1, groups=1, act=True):
        super().__init__()
        self.in_ch, self.out_ch, self.stride, self.groups = in_ch, out_ch, stride, groups
        self.has_identity = (in_ch == out_ch and stride == 1)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch))
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch))
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        if hasattr(self, 'fused_conv'):
            return self.act(self.fused_conv(x))
        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.has_identity:
            out = out + x
        return self.act(out)

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        w = conv.weight
        mean, var = bn.running_mean, bn.running_var
        gamma, beta, eps = bn.weight, bn.bias, bn.eps
        std = (var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return w * t, beta - mean * gamma / std

    def fuse(self):
        if hasattr(self, 'fused_conv'):
            return
        k3, b3 = self._fuse_conv_bn(self.branch_3x3[0], self.branch_3x3[1])
        k1, b1 = self._fuse_conv_bn(self.branch_1x1[0], self.branch_1x1[1])
        kernel = k3 + F.pad(k1, [1, 1, 1, 1])
        bias = b3 + b1
        if self.has_identity:
            k_id = torch.zeros_like(kernel)
            for i in range(self.in_ch):
                k_id[i, i % (self.in_ch // self.groups), 1, 1] = 1.0
            kernel = kernel + k_id
        self.fused_conv = nn.Conv2d(self.in_ch, self.out_ch, 3, self.stride, 1,
                                    groups=self.groups, bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        del self.branch_3x3, self.branch_1x1


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation."""
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.pool(x))


class StripPoolingAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.Sigmoid())

    def forward(self, x):
        h_strip = x.mean(dim=3, keepdim=True)
        w_strip = x.mean(dim=2, keepdim=True)
        return x * self.gate_conv(h_strip + w_strip)


class GlobalContextBlock(nn.Module):
    """GCNet-style global context."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.context_weight = nn.Conv2d(dim, 1, 1)
        hidden = max(dim // reduction, 8)
        self.transform = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        context_mask = self.context_weight(x).view(B, 1, H * W)
        context_mask = F.softmax(context_mask, dim=-1)
        context = torch.bmm(x.view(B, C, H * W), context_mask.transpose(1, 2)).unsqueeze(-1)
        return x + self.transform(context)


class MinimalMultiScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.branch1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.branch2 = nn.Conv2d(dim, dim, 3, 1, 2, dilation=2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.bn(self.branch1(x) + self.branch2(x))


def _pick_groups(in_ch, out_ch, max_g=4):
    for g in (max_g, 2, 1):
        if in_ch % g == 0 and out_ch % g == 0:
            return g
    return 1


class MinimalCrossScale(nn.Module):
    def __init__(self, dim_high, dim_low):
        super().__init__()
        self.low_to_high = nn.Conv2d(dim_low, dim_high, 1,
                                     groups=_pick_groups(dim_low, dim_high), bias=False)
        self.high_to_low = nn.Conv2d(dim_high, dim_low, 1,
                                     groups=_pick_groups(dim_high, dim_low), bias=False)

    def forward(self, x_high, x_low):
        low_up = F.interpolate(self.low_to_high(x_low), size=x_high.shape[2:], mode='nearest')
        high_down = F.adaptive_avg_pool2d(self.high_to_low(x_high), x_low.shape[2:])
        return x_high + low_up * 0.3, x_low + high_down * 0.3


class LightweightSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_hidden = c1 // 4
        self.cv1 = ConvBN(c1, c_hidden, 1)
        self.cv2 = ConvBN(c_hidden * 4, c2, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class UltraLightFusion(nn.Module):
    def __init__(self, high_ch, low_ch, out_ch):
        super().__init__()
        self.proj_high = nn.Conv2d(high_ch, out_ch, 1,
                                   groups=_pick_groups(high_ch, out_ch), bias=False)
        self.proj_low = nn.Conv2d(low_ch, out_ch, 1,
                                  groups=_pick_groups(low_ch, out_ch), bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_high, x_low):
        x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        return self.act(self.bn(self.proj_high(x_high) + self.proj_low(x_low)))


class FastConvexUpsample(nn.Module):
    """Convex upsampling via pixel-shuffle masks (GPU path only)."""
    def __init__(self, feat_ch, scale=4, temperature=1.0):
        super().__init__()
        self.scale = scale
        self.temperature = temperature
        hidden = max(feat_ch // 4, 8)
        self.mask_pred = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 9 * scale * scale, 1))

    def forward(self, feat, depth):
        B, _, H, W = depth.shape
        S = self.scale
        mask = self.mask_pred(feat).view(B, 9, S * S, H, W)
        mask = F.softmax(mask / self.temperature, dim=1)
        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        neighbors = F.unfold(depth_pad, 3).view(B, 9, 1, H, W)
        up = (mask * neighbors).sum(1)
        return F.relu(F.pixel_shuffle(up.view(B, S * S, H, W), S))


class ZipDepthEncoder(nn.Module):
    def __init__(self, in_ch, dims, depths):
        super().__init__()
        self.stem_half = ConvBN(in_ch, dims[0] // 2, k=3, s=2)
        self.stem_quarter = ConvBN(dims[0] // 2, dims[0], k=3, s=2)

        self.stage1 = nn.Sequential(*[QARepBlock(dims[0], dims[0]) for _ in range(depths[0])])

        self.down2 = QARepBlock(dims[0], dims[1], stride=2)
        self.stage2 = nn.Sequential(
            *[QARepBlock(dims[1], dims[1]) for _ in range(depths[1] - 1)],
            QARepBlock(dims[1], dims[1]), MinimalMultiScale(dims[1]),
            StripPoolingAttention(dims[1]))

        self.down3 = QARepBlock(dims[1], dims[2], stride=2)
        self.stage3 = nn.Sequential(
            *[QARepBlock(dims[2], dims[2]) for _ in range(depths[2] - 1)],
            QARepBlock(dims[2], dims[2]), ChannelAttention(dims[2]),
            GlobalContextBlock(dims[2]))

        self.down4 = QARepBlock(dims[2], dims[3], stride=2)
        self.stage4 = nn.Sequential(*[QARepBlock(dims[3], dims[3]) for _ in range(depths[3])])

        self.spp = LightweightSPPF(dims[3], dims[3])
        self.cross_scale = MinimalCrossScale(dims[2], dims[3])

    def forward(self, x):
        s_half = self.stem_half(x)
        s1 = self.stage1(self.stem_quarter(s_half))
        s2 = self.stage2(self.down2(s1))
        s3 = self.stage3(self.down3(s2))
        s4 = self.spp(self.stage4(self.down4(s3)))
        s3, s4 = self.cross_scale(s3, s4)
        return s_half, [s1, s2, s3, s4]

    def fuse(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'fuse'):
                m.fuse()


class ZipDepthDecoder(nn.Module):
    def __init__(self, enc_dims, half_ch, dec_ch, half_dec_ch):
        super().__init__()
        c1, c2, c3, c4 = enc_dims
        ch4, ch3, ch2, ch1 = dec_ch * 3, dec_ch * 2, int(dec_ch * 1.5), dec_ch
        self.proj4 = ConvBN(c4, ch4, 1)
        self.fuse3 = UltraLightFusion(c3, ch4, ch3)
        self.fuse2 = UltraLightFusion(c2, ch3, ch2)
        self.fuse1 = UltraLightFusion(c1, ch2, ch1)

        ch_half = half_dec_ch
        self.fuse_half = UltraLightFusion(high_ch=half_ch, low_ch=ch1, out_ch=ch_half)
        self.head_half = nn.Conv2d(ch_half, 1, 3, padding=1)
        self.convex_up = FastConvexUpsample(feat_ch=ch_half, scale=2)

    def forward(self, s_half, feats):
        c1, c2, c3, c4 = feats
        f4 = self.proj4(c4)
        f3 = self.fuse3(c3, f4)
        f2 = self.fuse2(c2, f3)
        f1 = self.fuse1(c1, f2)
        f_half = self.fuse_half(s_half, f1)
        return self.convex_up(f_half, self.head_half(f_half))


class ZipDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ZipDepthEncoder(in_ch=3, dims=DIMS, depths=DEPTHS)
        self.decoder = ZipDepthDecoder(enc_dims=DIMS, half_ch=DIMS[0] // 2,
                                       dec_ch=DEC_CH, half_dec_ch=HALF_DEC_CH)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        s_half, enc_feats = self.encoder((x - self.mean) / self.std)
        return self.decoder(s_half, enc_feats)

    def fuse_for_inference(self):
        self.eval()
        self.encoder.fuse()
        return self


if __name__ == "__main__":
    import os
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "checkpoints", "zipdepth_base.pth")
    model = ZipDepth()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    model.fuse_for_inference()
    y = model(torch.rand(1, 3, 384, 384))
    assert y.shape == (1, 1, 384, 384), y.shape
    print(f"OK: strict load + fused forward, output {tuple(y.shape)}")
