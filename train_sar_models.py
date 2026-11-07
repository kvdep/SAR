import os
import sys
import time
import math
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rasterio

# Настройка UTF-8 для вывода в консоль Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[ИНФОРМАЦИЯ] Используемое вычислительное устройство: {device}", flush=True)
if torch.cuda.is_available():
    print(f"[ИНФОРМАЦИЯ] Графический процессор: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[ИНФОРМАЦИЯ] Доступная видеопамять VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} ГБ", flush=True)

# =========================================================================
# 1. СТАТИСТИЧЕСКИЕ ФУНКЦИИ ПОТЕРЬ
# =========================================================================

class SSIMLoss(nn.Module):
    """Дифференцируемый индекс структурного подобия SSIM (Wang et al., 2004)"""
    def __init__(self, window_size: int = 11, channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.pad = window_size // 2
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        
        k = self.kernel.to(pred.device)
        mu1 = F.conv2d(pred, k, padding=self.pad, groups=self.channels)
        mu2 = F.conv2d(target, k, padding=self.pad, groups=self.channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, k, padding=self.pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, k, padding=self.pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, k, padding=self.pad, groups=self.channels) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-10)
        return 1.0 - ssim_map.mean()

def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Штраф полной вариации (Total Variation Loss)"""
    diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()

# =========================================================================
# 2. АРХИТЕКТУРЫ НЕЙРОННЫХ СЕТЕЙ
# =========================================================================

# --- МОДЕЛЬ 1: SAR-ResNet (Vitale et al., 2021, arXiv:2104.09350v2) ---
class SAR_ResNet(nn.Module):
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        residual_speckle = self.net(y)
        x_hat = torch.clamp(y - residual_speckle, min=1e-6)
        return x_hat


# --- МОДЕЛЬ 2: MERLIN CNN (Complex Self-Supervised Despeckling, Dalsasso et al., IEEE TGRS 2021) ---
class SAR_MERLIN_CNN(nn.Module):
    """Каноническая самообучающаяся архитектура MERLIN CNN (Dalsasso et al., 2021):
    Полноразмерная сверточная сеть с каскадом дилатированных слоев (d=1, 2, 3, 5, 2, 1)
    и лог-резидуальным вычитанием спекл-компоненты. Сохраняет точечные отражатели,
    крутые склоны гор и дюны пустынь без размытия и синих артефактов."""
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

# Каноническая дилатированная модель
MERLIN_DenseDilated_CNN = SAR_MERLIN_CNN

# Импорт дополнительных исследованных архитектур
from explore_merlin_architectures import (
    MERLIN_AttnUNet_CBAM,
    MERLIN_ResUNet_Deep,
    MERLIN_UNet_Base
)

class SAR_MERLIN_BlindSpot_Legacy(nn.Module):
    """Исходная слепая сеть с маскированием центрального отсчета и функцией Softplus."""
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
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        out = self.fuse(torch.cat([d1, e1], dim=1))
        return torch.clamp(out, min=1e-6)

# Алиас для обратной совместимости
SAR_MERLIN_BlindSpot = SAR_MERLIN_CNN


# --- МОДЕЛЬ 3: SAR-Restormer / Transformer (Perera et al., 2022) ---
class MDTA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
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

class GDFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.dwconv(self.project_in(x))
        x1, x2 = x_proj.chunk(2, dim=1)
        x_gated = F.gelu(x1) * x2
        return self.project_out(x_gated)

class RestormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = GDFN(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SAR_Restormer(nn.Module):
    def __init__(self, in_channels: int = 1, base_dim: int = 32):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        self.enc1 = nn.Sequential(*[RestormerBlock(base_dim, num_heads=2) for _ in range(2)])
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[RestormerBlock(base_dim * 2, num_heads=4) for _ in range(2)])
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.reduce = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1)
        self.dec1 = nn.Sequential(*[RestormerBlock(base_dim, num_heads=2) for _ in range(2)])
        self.proj_out = nn.Sequential(
            nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1),
            nn.Softplus()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(x)
        e1 = self.enc1(feat)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        u1 = self.up1(e2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        cat1 = self.reduce(torch.cat([u1, e1], dim=1))
        dec = self.dec1(cat1)
        out = self.proj_out(dec)
        return torch.clamp(out, min=1e-6)

# Экспорт исследованных и модернизированных архитектур
from run_massive_sar_dl_experiments import (
    SAR_YOLO26,
    SAR_Restormer_V2,
    MERLIN_Scale_Study,
    SAR_ResNet_Study
)

# Перенаправление SAR_Restormer на стабильную архитектуру V2
SAR_Restormer_Legacy = SAR_Restormer
SAR_Restormer = SAR_Restormer_V2


# =========================================================================
# 3. ВЫСОКОСКОРОСТНАЯ ГЕНЕРАЦИЯ ОБУЧАЮЩИХ ВЫБОРОК В ПАМЯТИ
# =========================================================================

def generate_multi_terrain_tensor(num_samples: int = 600, patch_size: int = 128, looks: float = 4.4, seed: int = 42):
    print(f"[ГЕНЕРАЦИЯ] Создание {num_samples} синтетических патчей {patch_size}x{patch_size} с L={looks}...", flush=True)
    np.random.seed(seed)
    P = patch_size
    gt_list = []
    
    for i in range(num_samples):
        terrain = i % 5
        gt = np.zeros((P, P), dtype=np.float32)
        if terrain == 0:
            gt[:] = 0.015 + 0.01 * np.random.rand()
        elif terrain == 1:
            sr, sc = np.random.randint(24, 48), np.random.randint(24, 48)
            for r in range(0, P, sr):
                for c in range(0, P, sc):
                    gt[r:r+sr, c:c+sc] = 0.08 + 0.25 * np.random.rand()
                    gt[r:r+2, c:c+sc] = 0.03
                    gt[r:r+sr, c:c+2] = 0.03
        elif terrain == 2:
            y, x = np.mgrid[0:P, 0:P]
            freq = 10.0 + 8.0 * np.random.rand()
            base = 0.12 + 0.04 * np.sin(x / freq) * np.cos(y / freq)
            tex = ndimage.gaussian_filter(np.random.randn(P, P).astype(np.float32) * 0.03, sigma=1.8)
            gt[:] = np.clip(base + tex, 0.03, 0.35)
        elif terrain == 3:
            gt[:] = 0.06
            step = np.random.randint(28, 36)
            for s in range(8, P, step):
                gt[s-2:s+2, :] = 0.02
                gt[:, s-2:s+2] = 0.02
            for r in range(4, P - 16, step):
                for c in range(4, P - 16, step):
                    gt[r:r+14, c:c+14] = 0.25 + 0.20 * np.random.rand()
                    if np.random.rand() > 0.5:
                        gt[r+2, c+2] = 3.5
        else:
            y, x = np.mgrid[0:P, 0:P]
            angle = np.random.rand() * math.pi
            grad = (x * np.cos(angle) + y * np.sin(angle)) / P
            gt[:] = np.clip(0.05 + 0.30 * (grad + 0.15 * np.sin(grad * 8.0)), 0.02, 0.45)
            
        gt_list.append(gt)
        
    gt_arr = np.stack(gt_list, axis=0)[:, np.newaxis, :, :]
    speckle_arr = np.random.gamma(shape=looks, scale=1.0 / looks, size=gt_arr.shape).astype(np.float32)
    noisy_arr = gt_arr * speckle_arr
    
    gt_tensor = torch.from_numpy(gt_arr).to(device)
    noisy_tensor = torch.from_numpy(noisy_arr).to(device)
    print(f"[УСПЕХ] Синтетический датасет сгенерирован: {gt_tensor.shape} на GPU", flush=True)
    return noisy_tensor, gt_tensor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TIF_PATH = os.path.join(BASE_DIR, "HTML Визуализация", "данные", "sar_2025_0_0.tif")
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "temp", "data_cache")
DEFAULT_WEIGHTS_DIR = os.path.join(BASE_DIR, "HTML Визуализация", "веса")


def load_real_sar_tensor(num_patches: int = 600, patch_size: int = 128, tif_path: str = None):
    if tif_path is None:
        tif_path = DEFAULT_TIF_PATH
    print(f"[ГЕНЕРАЦИЯ] Извлечение {num_patches} реальных патчей Sentinel-1 из {tif_path}...", flush=True)
    patches = []
    if os.path.exists(tif_path):
        with rasterio.open(tif_path) as src:
            H, W = src.shape
            attempts = 0
            while len(patches) < num_patches and attempts < num_patches * 10:
                attempts += 1
                r = np.random.randint(150, H - patch_size - 150)
                c = np.random.randint(150, W - patch_size - 150)
                data = src.read(1, window=((r, r + patch_size), (c, c + patch_size)))
                if np.all(np.isfinite(data)) and np.all(data > 0):
                    p = data.astype(np.float32)
                    p = np.clip(p, 1e-4, np.percentile(p, 99.5))
                    patches.append(p)
                    
    if len(patches) < num_patches:
        cache_dir = DEFAULT_CACHE_DIR
        for name in ['city', 'forest', 'mountain', 'desert', 'water']:
            npz = os.path.join(cache_dir, f"raw_{name}.npz")
            if os.path.exists(npz):
                d = np.load(npz)
                vv = d['vv']
                for _ in range(50):
                    if len(patches) >= num_patches:
                        break
                    r = np.random.randint(0, vv.shape[0] - patch_size)
                    c = np.random.randint(0, vv.shape[1] - patch_size)
                    crop = vv[r:r+patch_size, c:c+patch_size].astype(np.float32)
                    if np.all(np.isfinite(crop)) and np.all(crop > 0):
                        patches.append(crop)
                        
    real_arr = np.stack(patches[:num_patches], axis=0)[:, np.newaxis, :, :]
    real_tensor = torch.from_numpy(real_arr).to(device)
    print(f"[УСПЕХ] Реальный датасет Sentinel-1 загружен: {real_tensor.shape} на GPU", flush=True)
    return real_tensor


# =========================================================================
# 4. ТРЕНИРОВОЧНЫЕ ЦИКЛЫ
# =========================================================================

def train_sar_resnet_fast(noisy_data: torch.Tensor, gt_data: torch.Tensor, epochs: int = 30, batch_size: int = 32) -> SAR_ResNet:
    print("\n" + "="*70, flush=True)
    print("ОБУЧЕНИЕ МОДЕЛИ 1: SAR-ResNet (Vitale et al., 2021) НА RTX 3080", flush=True)
    print("="*70, flush=True)
    
    model = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    scaler = torch.amp.GradScaler('cuda')
    ssim_crit = SSIMLoss(window_size=11, channels=1).to(device)
    
    N = noisy_data.shape[0]
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_b = 0
        
        for b in range(0, N, batch_size):
            idx = perm[b:b+batch_size]
            bx = noisy_data[idx]
            by = gt_data[idx]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(bx)
                l_mse = F.mse_loss(pred, by)
                l_ssim = ssim_crit(pred, by, max_val=1.0)
                l_tv = tv_loss(pred)
                loss = 1.0 * l_mse + 1.0 * l_ssim + 1e-5 * l_tv
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_b += 1
            
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            avg_l = total_loss / max(n_b, 1)
            print(f"SAR-ResNet | Эпоха [{epoch:02d}/{epochs:02d}] | LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {avg_l:.5f}", flush=True)
            
    print(f"[УСПЕХ] SAR-ResNet обучен за {time.time() - t_start:.2f} сек на {torch.cuda.get_device_name(0)}", flush=True)
    return model


def train_merlin_blindspot_fast(real_data: torch.Tensor, epochs: int = 30, batch_size: int = 32) -> SAR_MERLIN_BlindSpot:
    print("\n" + "="*70, flush=True)
    print("ОБУЧЕНИЕ МОДЕЛИ 2: MERLIN Blind-Spot CNN (Dalsasso et al.) НА RTX 3080", flush=True)
    print("="*70, flush=True)
    
    model = SAR_MERLIN_BlindSpot(in_channels=1, num_features=48).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    N, C, H, W = real_data.shape
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_b = 0
        
        for b in range(0, N, batch_size):
            idx = perm[b:b+batch_size]
            bx = real_data[idx].clone()
            
            # Маска слепых пятен (Noise2Void / Blind-Spot: 2.5% пикселей)
            mask = (torch.rand_like(bx) < 0.025)
            shift_x = torch.roll(bx, shifts=1, dims=2)
            bx_masked = torch.where(mask, shift_x, bx)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                r_hat = model(bx_masked)
                # Точное отрицательное Гамма-правдоподобие на скрытых пикселях
                nll = torch.log(r_hat) + (bx / r_hat)
                loss = (nll * mask).sum() / (mask.sum() + 1e-8)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_b += 1
            
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            avg_l = total_loss / max(n_b, 1)
            print(f"MERLIN Blind-Spot | Эпоха [{epoch:02d}/{epochs:02d}] | LR: {scheduler.get_last_lr()[0]:.6f} | Blind-Spot NLL: {avg_l:.5f}", flush=True)
            
    print(f"[УСПЕХ] MERLIN обучен за {time.time() - t_start:.2f} сек на {torch.cuda.get_device_name(0)}", flush=True)
    return model


def train_sar_transformer_fast(noisy_data: torch.Tensor, gt_data: torch.Tensor, epochs: int = 30, batch_size: int = 16) -> SAR_Restormer:
    print("\n" + "="*70, flush=True)
    print("ОБУЧЕНИЕ МОДЕЛИ 3: SAR-Restormer / Transformer (Perera et al., 2022) НА RTX 3080", flush=True)
    print("="*70, flush=True)
    
    model = SAR_Restormer(in_channels=1, base_dim=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    ssim_crit = SSIMLoss(window_size=11, channels=1).to(device)
    
    N = noisy_data.shape[0]
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_b = 0
        
        for b in range(0, N, batch_size):
            idx = perm[b:b+batch_size]
            bx = noisy_data[idx]
            by = gt_data[idx]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(bx)
                l1 = F.l1_loss(pred, by)
                l_ssim = ssim_crit(pred, by, max_val=1.0)
                l_tv = tv_loss(pred)
                loss = 1.0 * l1 + 0.5 * l_ssim + 5e-5 * l_tv
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_b += 1
            
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            avg_l = total_loss / max(n_b, 1)
            print(f"SAR-Restormer | Эпоха [{epoch:02d}/{epochs:02d}] | LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {avg_l:.5f}", flush=True)
            
    print(f"[УСПЕХ] SAR-Restormer обучен за {time.time() - t_start:.2f} сек на {torch.cuda.get_device_name(0)}", flush=True)
    return model


# =========================================================================
# 5. ГЛАВНАЯ ФУНКЦИЯ
# =========================================================================

def main():
    weights_dir = DEFAULT_WEIGHTS_DIR
    os.makedirs(weights_dir, exist_ok=True)
    
    # 1. Генерация данных
    noisy_synth, gt_synth = generate_multi_terrain_tensor(num_samples=600, patch_size=128, looks=4.4, seed=42)
    real_sar = load_real_sar_tensor(num_patches=600, patch_size=128)
    
    # 2. Обучение Модели 1: SAR-ResNet (если не сохранена)
    path_resnet = os.path.join(weights_dir, "sar_resnet_vitale.pth")
    if not os.path.exists(path_resnet):
        m_resnet = train_sar_resnet_fast(noisy_synth, gt_synth, epochs=30, batch_size=32)
        torch.save(m_resnet.state_dict(), path_resnet)
        print(f"[СОХРАНЕНО] Веса SAR-ResNet: {path_resnet}", flush=True)
    else:
        print(f"[ПРОПУСК] Веса SAR-ResNet уже обучены и существуют: {path_resnet}", flush=True)
        
    # 3. Обучение Модели 2: MERLIN Blind-Spot
    path_merlin = os.path.join(weights_dir, "sar_merlin_blindspot.pth")
    m_merlin = train_merlin_blindspot_fast(real_sar, epochs=30, batch_size=32)
    torch.save(m_merlin.state_dict(), path_merlin)
    print(f"[СОХРАНЕНО] Веса MERLIN: {path_merlin}", flush=True)
    
    # 4. Обучение Модели 3: SAR-Restormer
    path_transformer = os.path.join(weights_dir, "sar_restormer.pth")
    m_transformer = train_sar_transformer_fast(noisy_synth, gt_synth, epochs=30, batch_size=16)
    torch.save(m_transformer.state_dict(), path_transformer)
    print(f"[СОХРАНЕНО] Веса SAR-Restormer: {path_transformer}", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("ВСЕ 3 МОДЕЛИ ГЛУБОКОГО ОБУЧЕНИЯ УСПЕШНО ОБУЧЕНЫ И СОХРАНЕНЫ!", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
