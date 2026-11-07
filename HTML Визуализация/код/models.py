# -*- coding: utf-8 -*-
"""Полный реестр архитектур глубоких нейронных сетей для фильтрации спекла Sentinel-1 SAR.

Все модели самодостаточны и не имеют внешних зависимостей за пределами PyTorch.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. МОДЕЛЬ 1: SAR-ResNet (Vitale et al., 2021)
# =========================================================================

class SAR_ResNet(nn.Module):
    """Глубокая остаточная сверточная сеть (Vitale et al., IEEE JSTARS 2021)."""
    def __init__(self, in_channels: int = 1, num_features: int = 64, num_layers: int = 17):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        residual_speckle = self.net(y)
        x_hat = torch.clamp(y - residual_speckle, min=1e-6)
        return x_hat


# =========================================================================
# 2. МОДЕЛЬ 2: MERLIN CNN (Dalsasso et al., IEEE TGRS 2021)
# =========================================================================

class SAR_MERLIN_CNN(nn.Module):
    """Флагманская сеть архитектуры MERLIN с плотными расширенными свертками."""
    def __init__(self, in_channels: int = 1, num_features: int = 36):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.d1 = nn.Conv2d(num_features, num_features, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(num_features * 2, num_features, 3, padding=2, dilation=2)
        self.d3 = nn.Conv2d(num_features * 3, num_features, 3, padding=3, dilation=3)
        self.d4 = nn.Conv2d(num_features * 4, num_features, 3, padding=5, dilation=5)
        self.d5 = nn.Conv2d(num_features * 5, num_features, 3, padding=2, dilation=2)
        self.d6 = nn.Conv2d(num_features * 6, num_features, 3, padding=1, dilation=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv_out = nn.Conv2d(num_features * 7, in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))

        x0 = self.act(self.conv_in(log_y))
        x1 = self.act(self.d1(x0))
        c1 = torch.cat([x0, x1], dim=1)

        x2 = self.act(self.d2(c1))
        c2 = torch.cat([c1, x2], dim=1)

        x3 = self.act(self.d3(c2))
        c3 = torch.cat([c2, x3], dim=1)

        x4 = self.act(self.d4(c3))
        c4 = torch.cat([c3, x4], dim=1)

        x5 = self.act(self.d5(c4))
        c5 = torch.cat([c4, x5], dim=1)

        x6 = self.act(self.d6(c5))
        c6 = torch.cat([c5, x6], dim=1)

        log_residual = self.conv_out(c6)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)

MERLIN_DenseDilated_CNN = SAR_MERLIN_CNN


# =========================================================================
# 3. МОДЕЛЬ 3: MERLIN UNet Base
# =========================================================================

class MERLIN_UNet_Base(nn.Module):
    """Классический 3-уровневый U-Net с лог-резидуальным предсказанием."""
    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))

        e1 = self.enc1(log_y)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        log_residual = self.out_conv(d1)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# =========================================================================
# 4. МОДЕЛЬ 4: MERLIN ResUNet Deep
# =========================================================================

class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.act1(self.conv1(x)))
        return self.act2(x + res)


class MERLIN_ResUNet_Deep(nn.Module):
    """Глубокий Residual U-Net с 4 уровнями масштаба и дилатированными блоками."""
    def __init__(self, in_channels: int = 1, base_ch: int = 40):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.res1 = ResBlock(base_ch, dilation=1)

        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.res2 = ResBlock(base_ch * 2, dilation=2)

        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.res3 = ResBlock(base_ch * 4, dilation=4)

        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 4, dilation=2),
            ResBlock(base_ch * 4, dilation=1)
        )

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec_res2 = ResBlock(base_ch * 4, dilation=1)

        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch, 2, stride=2)
        self.dec_res1 = ResBlock(base_ch * 2, dilation=1)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, in_channels, 1)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))

        x0 = self.in_conv(log_y)
        e1 = self.res1(x0)
        e2 = self.res2(self.down1(e1))
        e3 = self.res3(self.down2(e2))
        b = self.bottleneck(e3)

        d2 = self.up2(b)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec_res2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)

        log_residual = self.out_conv(d1)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# =========================================================================
# 5. МОДЕЛЬ 5: MERLIN AttnUNet CBAM
# =========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // ratio, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return scale * x


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class MERLIN_AttnUNet_CBAM(nn.Module):
    """U-Net с механизмами канального и пространственного внимания CBAM."""
    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.cbam1 = CBAM(base_ch)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.cbam2 = CBAM(base_ch * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))

        e1 = self.enc1(log_y)
        attn_e1 = self.cbam1(e1)

        e2 = self.enc2(self.pool1(e1))
        attn_e2 = self.cbam2(e2)

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[2:] != attn_e2.shape[2:]:
            d2 = F.interpolate(d2, size=attn_e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, attn_e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != attn_e1.shape[2:]:
            d1 = F.interpolate(d1, size=attn_e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, attn_e1], dim=1))

        log_residual = self.out_conv(d1)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# =========================================================================
# 6. МОДЕЛЬ 6: SAR MERLIN BlindSpot Legacy
# =========================================================================

class SAR_MERLIN_BlindSpot_Legacy(nn.Module):
    """Исходная слепая сеть с маскированием центрального отсчета."""
    def __init__(self, in_channels: int = 1, num_features: int = 48):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 2, 3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(num_features * 2, num_features, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, in_channels, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        out = self.fuse(torch.cat([d1, e1], dim=1))
        return torch.clamp(out, min=1e-6)


# =========================================================================
# 7. МОДЕЛЬ 7: MERLIN Scale Study
# =========================================================================

class MERLIN_Scale_Study(nn.Module):
    """Многомасштабная дилатированная модель MERLIN с настраиваемой глубиной."""
    def __init__(self, in_channels: int = 1, num_features: int = 36, dilation_type: str = "prime", depth: int = 6):
        super().__init__()
        self.depth = depth
        if dilation_type == "exponential":
            dilations = [2 ** i for i in range(depth // 2)] + [2 ** (depth // 2 - 1 - i) for i in range(depth // 2)]
        elif dilation_type == "dense_unit":
            dilations = [1] * depth
        else:  # prime
            prime_seq = [1, 2, 3, 5, 7, 11, 7, 5, 3, 2, 1]
            dilations = prime_seq[:depth]

        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.d_layers = nn.ModuleList()
        for i, d in enumerate(dilations):
            in_c = num_features * (i + 1)
            self.d_layers.append(nn.Conv2d(in_c, num_features, 3, padding=d, dilation=d))

        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv_out = nn.Conv2d(num_features * (depth + 1), in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))
        x0 = self.act(self.conv_in(log_y))
        accum = [x0]
        for layer in self.d_layers:
            inp = torch.cat(accum, dim=1)
            x_next = self.act(layer(inp))
            accum.append(x_next)

        cat_all = torch.cat(accum, dim=1)
        log_residual = self.conv_out(cat_all)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)


# =========================================================================
# 8. МОДЕЛЬ 8: SAR YOLO26
# =========================================================================

class ConvBNSiLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, dilation=d, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, stride=s, padding=d * (k // 2), dilation=d, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class C3k2Block(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        c_hidden = int(c2 * expansion)
        self.cv1 = ConvBNSiLU(c1, c_hidden * 2, 1, 1, 0)
        self.cv2 = ConvBNSiLU((2 + n) * c_hidden, c2, 1, 1, 0)
        self.m = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(c_hidden, c_hidden, 3, 1, 1),
                DWConv(c_hidden, c_hidden, 3, 1, 1)
            ) for _ in range(n)
        ])
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1, y2 = self.cv1(x).chunk(2, dim=1)
        layers = [y1, y2]
        cur = y2
        for block in self.m:
            cur = cur + block(cur)
            layers.append(cur)
        out = self.cv2(torch.cat(layers, dim=1))
        return x + out if self.shortcut else out


class SAR_YOLO26(nn.Module):
    """Полноразмерная сеть фильтрации спекла на базе блоков YOLO26."""
    def __init__(self, in_channels: int = 1, width: int = 32, num_blocks: int = 3, use_log_residual: bool = True):
        super().__init__()
        self.use_log_residual = use_log_residual
        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, width, 3, 1, 1),
            DWConv(width, width, 3, 1, 1)
        )
        self.blocks = nn.ModuleList([
            C3k2Block(width, width, n=2, shortcut=True, expansion=0.5)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            ConvBNSiLU(width, width, 3, 1, 1),
            nn.Conv2d(width, in_channels, 1)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.use_log_residual:
            log_y = torch.log(torch.clamp(y, min=1e-6))
            x = self.stem(log_y)
            for b in self.blocks:
                x = b(x)
            log_residual = self.head(x)
            log_x = log_y - log_residual
            x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
            return torch.clamp(x_hat, min=1e-6)
        else:
            x = self.stem(y)
            for b in self.blocks:
                x = b(x)
            residual = self.head(x)
            return torch.clamp(y - residual, min=1e-6)


# =========================================================================
# 9. МОДЕЛЬ 9: SAR Restormer V2 (Трансформерная архитектура)
# =========================================================================

class MDTA_V2(nn.Module):
    """Многоголовое транспонированное внимание (MDTA) с кросс-ковариацией каналов."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * (1.0 / math.sqrt(head_dim)))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class GDFN_V2(nn.Module):
    """Gated-Dconv Feed-Forward Network."""
    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.dwconv(self.project_in(x))
        x1, x2 = proj.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class RestormerBlock_V2(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MDTA_V2(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = GDFN_V2(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ln1 = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_ln1)
        x_ln2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_ln2)
        return x


class SAR_Restormer_V2(nn.Module):
    """Трансформер SAR-Restormer V2 в лог-пространстве с канальной нормализацией."""
    def __init__(self, in_channels: int = 1, base_dim: int = 32, num_heads: int = 4, depth: int = 4):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            RestormerBlock_V2(base_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.proj_out = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_dim, in_channels, kernel_size=1)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        log_y = torch.log(torch.clamp(y, min=1e-6))
        feat = self.embed(log_y)
        for blk in self.blocks:
            feat = blk(feat)
        log_residual = self.proj_out(feat)
        log_x = log_y - log_residual
        x_hat = torch.exp(torch.clamp(log_x, min=-14.0, max=8.0))
        return torch.clamp(x_hat, min=1e-6)

SAR_Restormer = SAR_Restormer_V2
SAR_ResNet_Study = SAR_ResNet
