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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[ИНФОРМАЦИЯ] Устройство: {device} ({torch.cuda.get_device_name(0)})", flush=True)

# 1. Архитектура SAR-ResNet по Витале с правильным расчетом невязки
class SAR_ResNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_features: int = 64, num_layers: int = 17):
        super().__init__()
        layers = [nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # Предсказание невязки шума и вычитание
        residual_speckle = self.net(y)
        return torch.clamp(y - residual_speckle, min=1e-6)

# Функция потерь SSIM
class SSIMLoss(nn.Module):
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        k = self.kernel.to(pred.device)
        mu1 = F.conv2d(pred, k, padding=self.pad, groups=self.channels)
        mu2 = F.conv2d(target, k, padding=self.pad, groups=self.channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, k, padding=self.pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, k, padding=self.pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, k, padding=self.pad, groups=self.channels) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-10)
        return 1.0 - ssim_map.mean()

def train_resnet():
    print("\n" + "="*70, flush=True)
    print("ПЕРЕОБУЧЕНИЕ SAR-ResNet С ПРЯМЫМ ШТРАФОМ НЕВЯЗКИ И НОРМАЛИЗАЦИЕЙ", flush=True)
    print("="*70, flush=True)

    np.random.seed(42)
    N = 600
    P = 128
    looks = 4.4
    gt_list = []

    for i in range(N):
        t = i % 4
        gt = np.zeros((P, P), dtype=np.float32)
        if t == 0:  # Лесная текстура (гауссовские флуктуации полога)
            base = 0.08 + 0.04 * np.random.rand()
            tex = ndimage.gaussian_filter(np.random.randn(P, P).astype(np.float32) * 0.02, sigma=1.5)
            gt[:] = np.clip(base + tex, 0.02, 0.20)
        elif t == 1:  # Вода / однородная гладь
            gt[:] = 0.01 + 0.005 * np.random.rand()
        elif t == 2:  # Сельскохозяйственные поля (четкие прямоугольные границы)
            gt[:] = 0.05
            step = np.random.randint(24, 48)
            for r in range(0, P, step):
                for c in range(0, P, step):
                    gt[r:r+step, c:c+step] = 0.04 + 0.20 * np.random.rand()
        else:  # Город / яркие точечные цели
            gt[:] = 0.06
            for _ in range(6):
                r, c = np.random.randint(10, P-10), np.random.randint(10, P-10)
                gt[r:r+4, c:c+4] = 0.35 + 0.30 * np.random.rand()
                gt[r+2, c+2] = 2.0
        gt_list.append(gt)

    gt_arr = np.stack(gt_list)[:, None, :, :]
    speckle_arr = np.random.gamma(shape=looks, scale=1.0 / looks, size=gt_arr.shape).astype(np.float32)
    noisy_arr = gt_arr * speckle_arr

    # Нормализация в диапазон [0, 1] для предотвращения насыщения BatchNorm
    norm_factor = np.percentile(noisy_arr, 99.5)
    gt_norm = gt_arr / norm_factor
    noisy_norm = noisy_arr / norm_factor
    noise_residual = noisy_norm - gt_norm

    t_gt = torch.from_numpy(gt_norm).to(device)
    t_noisy = torch.from_numpy(noisy_norm).to(device)
    t_res_target = torch.from_numpy(noise_residual).to(device)

    model = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    ssim_crit = SSIMLoss().to(device)

    batch_size = 32
    t0 = time.time()
    for ep in range(1, 36):
        model.train()
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_b = 0

        for b in range(0, N, batch_size):
            idx = perm[b:b+batch_size]
            bx = t_noisy[idx]
            by_res = t_res_target[idx]
            by_clean = t_gt[idx]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_res = model.net(bx)
                pred_clean = torch.clamp(bx - pred_res, min=1e-6)

                # Композитный лосс: обучение невязки + сохранение чистого сигнала + SSIM
                l_res = F.l1_loss(pred_res, by_res) + 0.5 * F.mse_loss(pred_res, by_res)
                l_clean = F.l1_loss(pred_clean, by_clean)
                l_ssim = ssim_crit(pred_clean, by_clean)
                loss = 1.0 * l_res + 0.5 * l_clean + 0.5 * l_ssim

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        if ep % 5 == 0 or ep == 1:
            print(f"SAR-ResNet (Fixed) | Эпоха [{ep:02d}/35] | Loss: {total_loss/n_b:.5f}", flush=True)

    print(f"[УСПЕХ] SAR-ResNet обучен за {time.time() - t0:.2f} сек.", flush=True)

    # Сохранение весов
    weights_path = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\weights\sar_resnet_vitale.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"[СОХРАНЕНО] Обновленные веса: {weights_path}", flush=True)

    # Верификация на лесе
    cache_dir = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\data_cache"
    vv_forest = np.load(os.path.join(cache_dir, "raw_forest.npz"))['vv']

    model.eval()
    with torch.no_grad():
        scale_f = float(np.percentile(vv_forest, 99.5))
        t_f = torch.from_numpy((vv_forest / scale_f).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            out_norm = model(t_f)
        out_f = out_norm.squeeze().cpu().numpy() * scale_f

    enl_raw = float((vv_forest[150:250, 150:250].mean()**2) / (vv_forest[150:250, 150:250].var() + 1e-8))
    enl_fix = float((out_f[150:250, 150:250].mean()**2) / (out_f[150:250, 150:250].var() + 1e-8))
    diff_mean = float(np.abs(out_f - vv_forest).mean())

    print(f"\n[ВЕРИФИКАЦИЯ НА ЛЕСЕ МЕЩЁРЫ]")
    print(f"• Исходный ENL леса: {enl_raw:.2f}")
    print(f"• Новый ENL SAR-ResNet: {enl_fix:.2f} (прирост x{enl_fix/enl_raw:.1f})")
    print(f"• Средняя абсолютная дельта: {diff_mean:.5f}")
    print(f"• Идентичность устранена: {diff_mean > 0.005}", flush=True)

if __name__ == "__main__":
    train_resnet()
