import os
import sys
import time
import math
import json
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================
# 1. GROUND TRUTH BENCHMARK DATASET GENERATION
# =========================================================================

def generate_synthetic_sar_benchmark(size=512, looks=4.4, seed=42):
    """Генерация эталонного тестового полигона (Ground Truth) с 4 типами ландшафта:
    1. Однородная вода (темная гладкая область, backscatter = 0.02)
    2. Сельхозполя (кусочно-гладкие участки с четкими межами, backscatter = 0.15 - 0.35)
    3. Лесной массив (плавная пространственная текстура биомассы, backscatter = 0.08 - 0.22)
    4. Городской квартал (ортогональная сетка улиц и высокоинтенсивные точечные отражатели до 5.0)
    """
    np.random.seed(seed)
    gt = np.zeros((size, size), dtype=np.float32)
    
    # Квадрант 1: Вода (однородная) [верхний левый]
    gt[:size//2, :size//2] = 0.02
    
    # Квадрант 2: Поля [верхний правый]
    # Разбиение на прямоугольные участки
    for r in range(0, size//2, 64):
        for c in range(size//2, size, 64):
            val = 0.12 + 0.20 * np.random.rand()
            gt[r:r+64, c:c+64] = val
            # Межа между полями (тонкие темные линии)
            gt[r:r+2, c:c+64] = 0.04
            gt[r:r+64, c:c+2] = 0.04
            
    # Квадрант 3: Лес [нижний левый]
    # Плавные гармоники + стохастическая флуктуация полога
    y, x = np.mgrid[0:size//2, 0:size//2]
    forest_base = 0.10 + 0.04 * np.sin(x / 18.0) * np.cos(y / 18.0)
    forest_texture = ndimage.gaussian_filter(np.random.randn(size//2, size//2).astype(np.float32) * 0.03, sigma=2.0)
    gt[size//2:, :size//2] = np.clip(forest_base + forest_texture, 0.04, 0.25)
    
    # Квадрант 4: Город [нижний правый]
    # Темный асфальт дорог, средние крыши, точечные яркие цели
    gt[size//2:, size//2:] = 0.08
    # Сетка улиц
    for step in range(size//2 + 32, size, 48):
        gt[step-4:step+4, size//2:] = 0.03  # широкие проспекты
        gt[size//2:, step-4:step+4] = 0.03
    # Здания (прямоугольные блоки)
    for r in range(size//2 + 8, size - 32, 48):
        for c in range(size//2 + 8, size - 32, 48):
            gt[r:r+28, c:c+28] = 0.45 + 0.15 * np.random.rand()
            # Угловой отражатель (corner reflector) на углу здания
            gt[r+2, c+2] = 4.50
            
    # Наложение точного мультипликативного спекла Gamma(L, 1/L)
    # E[u] = 1.0, Var(u) = 1/L
    shape_param = looks
    scale_param = 1.0 / looks
    speckle = np.random.gamma(shape=shape_param, scale=scale_param, size=(size, size)).astype(np.float32)
    
    noisy = gt * speckle
    
    # Маска строго однородной зоны для вычисления ENL (внутри воды)
    homo_mask = np.zeros((size, size), dtype=bool)
    homo_mask[32:size//2-32, 32:size//2-32] = True
    
    return gt, noisy, homo_mask


# =========================================================================
# 2. МЕТРИКИ ОЦЕНКИ
# =========================================================================

def compute_benchmark_metrics(gt: np.ndarray, noisy: np.ndarray, filtered: np.ndarray, homo_mask: np.ndarray, looks=4.4):
    """Вычисление полного набора метрик из литературы:
    - Full-reference: PSNR, SSIM, MAE, EPI_GT
    - No-reference: ENL_homo, Ratio_Mean, Ratio_Var, Ratio_Moran
    """
    eps = 1e-4
    filt = np.maximum(filtered, eps)
    
    # 1. PSNR (dB)
    mse = np.mean((gt - filt) ** 2)
    max_val = np.max(gt)
    psnr = 10.0 * np.log10((max_val ** 2) / max(mse, 1e-12))
    
    # 2. MAE
    mae = np.mean(np.abs(gt - filt))
    
    # 3. SSIM (Wang et al., 2004)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    mu1 = ndimage.uniform_filter(gt, size=11)
    mu2 = ndimage.uniform_filter(filt, size=11)
    sig1_sq = ndimage.uniform_filter(gt**2, size=11) - mu1**2
    sig2_sq = ndimage.uniform_filter(filt**2, size=11) - mu2**2
    sig12 = ndimage.uniform_filter(gt * filt, size=11) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sig12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sig1_sq + sig2_sq + c2) + 1e-12)
    ssim_val = float(np.mean(ssim_map))
    
    # 4. EPI relative to Ground Truth (Edge Preservation Index)
    l_gt = ndimage.laplace(gt).flatten()
    l_f = ndimage.laplace(filt).flatten()
    denom = np.sqrt(np.sum(l_gt**2) * np.sum(l_f**2)) + 1e-12
    epi_gt = float(np.sum(l_gt * l_f) / denom)
    
    # 5. ENL in homogeneous area
    h_vals = filt[homo_mask]
    enl = float((np.mean(h_vals) ** 2) / (np.var(h_vals) + 1e-12))
    
    # 6. Ratio image statistics r = y / x_hat
    ratio = np.clip(noisy / filt, 0.0, 50.0)
    ratio_mean = float(np.mean(ratio))
    ratio_var = float(np.var(ratio))
    
    # 7. Spatial Moran's I on ratio image (spatial independence of residuals)
    sub = ratio[::4, ::4]
    sub_c = sub - np.mean(sub)
    v_sub = np.mean(sub_c**2) + 1e-12
    cov_x = np.mean(sub_c[:, :-1] * sub_c[:, 1:])
    cov_y = np.mean(sub_c[:-1, :] * sub_c[1:, :])
    moran_val = float(0.5 * (cov_x + cov_y) / v_sub)
    
    return {
        "psnr": float(round(psnr, 2)),
        "ssim": float(round(ssim_val, 4)),
        "mae": float(round(mae, 4)),
        "epi_gt": float(round(epi_gt, 4)),
        "enl_homo": float(round(enl, 2)),
        "ratio_mean": float(round(ratio_mean, 4)),
        "ratio_var": float(round(ratio_var, 4)),
        "ratio_moran": float(round(moran_val, 4))
    }


# =========================================================================
# 3. РЕАЛИЗАЦИЯ ВСЕХ ИССЛЕДУЕМЫХ МЕТОДОВ ИЗ ЛИТЕРАТУРЫ
# =========================================================================

# --- МЕТОД 1: Фильтр Ли (Lee Filter, MMSE) ---
def filter_lee(im: np.ndarray, w: int = 5, looks: float = 4.4) -> np.ndarray:
    Cu = 1.0 / math.sqrt(looks)
    m = ndimage.uniform_filter(im, size=w)
    sq = ndimage.uniform_filter(im**2, size=w)
    v = np.maximum(sq - m**2, 1e-10)
    ci = np.sqrt(v) / (m + 1e-9)
    # Коэффициент взвешивания Ли: b = max(0, 1 - Cu^2 / CI^2)
    b = np.clip(1.0 - (Cu**2) / (ci**2 + 1e-9), 0.0, 1.0)
    return m + b * (im - m)


# --- МЕТОД 2: Фильтр Фроста (Frost Filter, Adaptive Exponential) ---
def filter_frost(im: np.ndarray, w: int = 5, a: float = 2.0) -> np.ndarray:
    half = w // 2
    pad = np.pad(im, half, mode='reflect')
    out = np.zeros_like(im)
    
    # Координаты смещения внутри окна
    y, x = np.mgrid[-half:half+1, -half:half+1]
    dists = np.sqrt(x**2 + y**2)
    
    m = ndimage.uniform_filter(im, size=w)
    sq = ndimage.uniform_filter(im**2, size=w)
    v = np.maximum(sq - m**2, 1e-10)
    ci = np.sqrt(v) / (m + 1e-9)
    
    H, W = im.shape
    for i in range(H):
        for j in range(W):
            c_ij = ci[i, j]
            # Веса Фроста: exp(-a * CI * dist)
            weights = np.exp(-a * c_ij * dists)
            weights /= np.sum(weights)
            window = pad[i:i+w, j:j+w]
            out[i, j] = np.sum(window * weights)
    return out


# --- МЕТОД 3: Фильтр Куана (Kuan Filter, Non-stationary MMSE) ---
def filter_kuan(im: np.ndarray, w: int = 5, looks: float = 4.4) -> np.ndarray:
    Cu_sq = 1.0 / looks
    m = ndimage.uniform_filter(im, size=w)
    sq = ndimage.uniform_filter(im**2, size=w)
    v = np.maximum(sq - m**2, 1e-10)
    ci_sq = v / (m**2 + 1e-9)
    # Вес Куана: W = (1 - Cu^2 / CI^2) / (1 + Cu^2)
    w_weight = np.clip((1.0 - Cu_sq / (ci_sq + 1e-9)) / (1.0 + Cu_sq), 0.0, 1.0)
    return m + w_weight * (im - m)


# --- МЕТОД 4: Фильтр Gamma MAP (Bayesian MAP Lopes et al.) ---
def filter_gamma_map(im: np.ndarray, w: int = 5, looks: float = 4.4) -> np.ndarray:
    m = ndimage.uniform_filter(im, size=w)
    sq = ndimage.uniform_filter(im**2, size=w)
    v = np.maximum(sq - m**2, 1e-10)
    ci = np.sqrt(v) / (m + 1e-9)
    cu = 1.0 / math.sqrt(looks)
    cmax = math.sqrt(2.0) * cu
    out = np.copy(im)
    
    hetero = (ci > cu) & (ci < cmax)
    if np.any(hetero):
        ci_h = ci[hetero]
        im_h = im[hetero]
        m_h = m[hetero]
        a = (1.0 + cu**2) / (ci_h**2 - cu**2 + 1e-9)
        t = (a - looks - 1.0) * m_h
        d = t**2 + 4.0 * a * looks * im_h * m_h
        out[hetero] = (t + np.sqrt(np.maximum(d, 0.0))) / (2.0 * a)
    out[ci <= cu] = m[ci <= cu]
    return out


# --- МЕТОД 5: Нелокальный фильтр (SAR Non-Local Means / Patch-Based) ---
def filter_sar_nlm(im: np.ndarray, patch_rad: int = 2, search_rad: int = 6, h: float = 0.4) -> np.ndarray:
    H, W = im.shape
    pad_len = search_rad + patch_rad
    padded = np.pad(im, pad_len, mode='reflect')
    padded_log = np.log(np.maximum(padded, 1e-6))
    out = np.zeros_like(im)
    
    # Быстрый NLM через интегральные изображения или скользящее окно
    # Для детерминированного воспроизведения используем смещения в окрестности поиска
    weights_sum = np.zeros_like(im)
    pw = 2 * patch_rad + 1
    
    for dy in range(-search_rad, search_rad + 1, 2):
        for dx in range(-search_rad, search_rad + 1, 2):
            if dy == 0 and dx == 0:
                continue
            # Смещенный патч
            shifted = padded_log[pad_len + dy : pad_len + dy + H, pad_len + dx : pad_len + dx + W]
            center = padded_log[pad_len : pad_len + H, pad_len : pad_len + W]
            diff_sq = (center - shifted)**2
            patch_dist = ndimage.uniform_filter(diff_sq, size=pw)
            w_ij = np.exp(-patch_dist / (h**2))
            
            orig_shifted = padded[pad_len + dy : pad_len + dy + H, pad_len + dx : pad_len + dx + W]
            out += w_ij * orig_shifted
            weights_sum += w_ij
            
    # Самоподобие в центре
    out += 1.0 * im
    weights_sum += 1.0
    return out / weights_sum


# --- МЕТОД 6: Вейвлетная фильтрация в лог-области (Wavelet BayesShrink) ---
def filter_wavelet_sar(im: np.ndarray, threshold_scale: float = 1.0) -> np.ndarray:
    # 2-уровневый дискретный вейвлет-анализ Хаара / Добеши
    # Преобразование в лог-область с компенсацией смещения Эйлера: ln(I) - psi(L) + ln(L)
    L = 4.4
    log_bias = -0.117919
    v = np.log(np.maximum(im, 1e-6)) - log_bias
    
    # 2D вейвлет-разложение через фильтры Добеши db2
    # Реализация 2-уровневого DWT
    H, W = v.shape
    v_t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device)
    
    # Фильтры Хаара/Добеши
    h0 = torch.tensor([1.0, 1.0], device=device).view(1, 1, 1, 2) / math.sqrt(2.0)
    h1 = torch.tensor([1.0, -1.0], device=device).view(1, 1, 1, 2) / math.sqrt(2.0)
    
    # Разложение уровня 1
    ll = nn.functional.conv2d(nn.functional.conv2d(v_t, h0, stride=(1, 2)), h0.transpose(2, 3), stride=(2, 1))
    lh = nn.functional.conv2d(nn.functional.conv2d(v_t, h0, stride=(1, 2)), h1.transpose(2, 3), stride=(2, 1))
    hl = nn.functional.conv2d(nn.functional.conv2d(v_t, h1, stride=(1, 2)), h0.transpose(2, 3), stride=(2, 1))
    hh = nn.functional.conv2d(nn.functional.conv2d(v_t, h1, stride=(1, 2)), h1.transpose(2, 3), stride=(2, 1))
    
    # Теоретическая дисперсия лог-спекла: psi'(L) = psi'(4.4) = 0.255036
    sigma_noise = math.sqrt(0.255036)
    
    # BayesShrink пороговая обработка деталей: T = sigma_n^2 / sigma_x
    for band in [lh, hl, hh]:
        sig_band = torch.sqrt(torch.clamp(torch.mean(band**2) - sigma_noise**2, min=1e-6))
        T = (sigma_noise**2) / sig_band * threshold_scale
        # Мягкое пороговое сжатие: sign(x) * max(0, |x| - T)
        band.copy_(torch.sign(band) * torch.clamp(torch.abs(band) - T, min=0.0))
        
    # Обратное DWT уровня 1
    rec_h0 = h0.transpose(2, 3)
    rec_h1 = h1.transpose(2, 3)
    
    # Синтез через conv_transpose2d
    up_ll = nn.functional.conv_transpose2d(nn.functional.conv_transpose2d(ll, h0, stride=(1, 2)), rec_h0, stride=(2, 1))
    up_lh = nn.functional.conv_transpose2d(nn.functional.conv_transpose2d(lh, h0, stride=(1, 2)), rec_h1, stride=(2, 1))
    up_hl = nn.functional.conv_transpose2d(nn.functional.conv_transpose2d(hl, h1, stride=(1, 2)), rec_h0, stride=(2, 1))
    up_hh = nn.functional.conv_transpose2d(nn.functional.conv_transpose2d(hh, h1, stride=(1, 2)), rec_h1, stride=(2, 1))
    
    v_rec = (up_ll + up_lh + up_hl + up_hh).squeeze().cpu().numpy()
    # Возврат в интенсивность
    out = np.exp(v_rec)
    return out * (im.mean() / out.mean())


# --- МЕТОД 7: Total Variation ROF (Rudin-Osher-Fatemi, alpha=1.0) ---
def filter_tv_rof(im: np.ndarray, lam: float = 2.0, n_iter: int = 30) -> np.ndarray:
    im_t = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(device)
    I_clamped = torch.clamp(im_t, min=1e-6)
    v = torch.log(I_clamped)
    v_bar = v.clone()
    px = torch.zeros_like(I_clamped)
    py = torch.zeros_like(I_clamped)
    
    tau = 0.25
    sigma = 0.25
    
    for _ in range(n_iter):
        gx = nn.functional.pad(v_bar[:, :, :, 1:] - v_bar[:, :, :, :-1], (0, 1, 0, 0))
        gy = nn.functional.pad(v_bar[:, :, 1:, :] - v_bar[:, :, :-1, :], (0, 0, 0, 1))
        
        t_px = px + sigma * gx
        t_py = py + sigma * gy
        norm = torch.sqrt(t_px**2 + t_py**2 + 1e-12)
        denom = torch.clamp(norm, min=1.0)
        px = t_px / denom
        py = t_py / denom
        
        # Дивергенция
        dx = nn.functional.pad(px[:, :, :, :-1] - px[:, :, :, 1:], (1, 0, 0, 0))
        dy = nn.functional.pad(py[:, :, :-1, :] - py[:, :, 1:, :], (0, 0, 1, 0))
        div = dx + dy
        
        v_tilde = v - tau * div
        v_old = v.clone()
        for _ in range(3):
            exp_neg = torch.exp(-v)
            f_val = v - v_tilde + tau * lam * (1.0 - I_clamped * exp_neg)
            f_prime = 1.0 + tau * lam * I_clamped * exp_neg
            v = v - f_val / f_prime
        v_bar = 2.0 * v - v_old
        
    out = torch.exp(v).squeeze().cpu().numpy()
    return out * (im.mean() / out.mean())


# --- МЕТОД 8: Fractional TV (FTV, alpha=1.25, K=11) ---
def filter_ftv(im: np.ndarray, alpha: float = 1.25, lam: float = 4.0, n_iter: int = 25) -> np.ndarray:
    K = 11
    w = np.zeros(K, dtype=np.float32)
    w[0] = 1.0
    for k in range(1, K):
        w[k] = (1.0 - (alpha + 1.0) / k) * w[k - 1]
    w[0] = -np.sum(w[1:])
    
    w_x = torch.from_numpy(w).flip(0).view(1, 1, 1, K).to(device)
    w_y = torch.from_numpy(w).flip(0).view(1, 1, K, 1).to(device)
    w_x_adj = torch.from_numpy(w).view(1, 1, 1, K).to(device)
    w_y_adj = torch.from_numpy(w).view(1, 1, K, 1).to(device)
    
    L_est = 2.0 * math.sqrt(float(K))
    tau = 0.9 / L_est
    sigma = 0.9 / L_est
    
    im_t = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(device)
    I_clamped = torch.clamp(im_t, min=1e-6)
    v = torch.log(I_clamped)
    v_bar = v.clone()
    px = torch.zeros_like(I_clamped)
    py = torch.zeros_like(I_clamped)
    
    for _ in range(n_iter):
        pad_x = nn.functional.pad(v_bar, (K - 1, 0, 0, 0), mode="replicate")
        pad_y = nn.functional.pad(v_bar, (0, 0, K - 1, 0), mode="replicate")
        gx = nn.functional.conv2d(pad_x, w_x)
        gy = nn.functional.conv2d(pad_y, w_y)
        
        t_px = px + sigma * gx
        t_py = py + sigma * gy
        norm = torch.sqrt(t_px**2 + t_py**2 + 1e-12)
        denom = torch.clamp(norm, min=1.0)
        px = t_px / denom
        py = t_py / denom
        
        pad_px = nn.functional.pad(px, (0, K - 1, 0, 0), mode="constant", value=0.0)
        pad_py = nn.functional.pad(py, (0, 0, 0, K - 1), mode="constant", value=0.0)
        div = nn.functional.conv2d(pad_px, w_x_adj) + nn.functional.conv2d(pad_py, w_y_adj)
        
        v_tilde = v - tau * div
        v_old = v.clone()
        for _ in range(3):
            exp_neg = torch.exp(-v)
            f_val = v - v_tilde + tau * lam * (1.0 - I_clamped * exp_neg)
            f_prime = 1.0 + tau * lam * I_clamped * exp_neg
            v = v - f_val / f_prime
        v_bar = 2.0 * v - v_old
        
    out = torch.exp(v).squeeze().cpu().numpy()
    return out * (im.mean() / out.mean())


# --- МЕТОД 9: Joint FTV-SARP (Адаптивный по вариации CI) ---
def filter_ftv_sarp(im: np.ndarray, lam_base: float = 3.5, lam_boost: float = 8.0, n_iter: int = 25) -> np.ndarray:
    K = 11
    alpha = 1.25
    w = np.zeros(K, dtype=np.float32)
    w[0] = 1.0
    for k in range(1, K):
        w[k] = (1.0 - (alpha + 1.0) / k) * w[k - 1]
    w[0] = -np.sum(w[1:])
    
    w_x = torch.from_numpy(w).flip(0).view(1, 1, 1, K).to(device)
    w_y = torch.from_numpy(w).flip(0).view(1, 1, K, 1).to(device)
    w_x_adj = torch.from_numpy(w).view(1, 1, 1, K).to(device)
    w_y_adj = torch.from_numpy(w).view(1, 1, K, 1).to(device)
    
    L_est = 2.0 * math.sqrt(float(K))
    tau = 0.9 / L_est
    sigma = 0.9 / L_est
    
    im_t = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(device)
    I_clamped = torch.clamp(im_t, min=1e-6)
    
    # CFAR
    mu_15 = nn.functional.avg_pool2d(I_clamped, 15, stride=1, padding=7)
    sq_15 = nn.functional.avg_pool2d(I_clamped**2, 15, stride=1, padding=7)
    sig_15 = torch.sqrt(torch.clamp(sq_15 - mu_15**2, min=1e-8))
    cfar_mask = (I_clamped > mu_15 + 5.5 * sig_15) & (I_clamped > 0.5)
    
    # SARP modulation
    mu_7 = nn.functional.avg_pool2d(I_clamped, 7, stride=1, padding=3)
    sq_7 = nn.functional.avg_pool2d(I_clamped**2, 7, stride=1, padding=3)
    sig_7 = torch.sqrt(torch.clamp(sq_7 - mu_7**2, min=1e-8))
    ci_map = sig_7 / (mu_7 + 1e-8)
    Cu = 1.0 / math.sqrt(4.4)
    Cmax = math.sqrt(2.0 / 4.4)
    delta = torch.clamp((ci_map - Cu) / (Cmax - Cu + 1e-8), 0.0, 1.0)
    lam_map = lam_base + lam_boost * delta
    
    v = torch.log(I_clamped)
    v_bar = v.clone()
    px = torch.zeros_like(I_clamped)
    py = torch.zeros_like(I_clamped)
    
    for _ in range(n_iter):
        pad_x = nn.functional.pad(v_bar, (K - 1, 0, 0, 0), mode="replicate")
        pad_y = nn.functional.pad(v_bar, (0, 0, K - 1, 0), mode="replicate")
        gx = nn.functional.conv2d(pad_x, w_x)
        gy = nn.functional.conv2d(pad_y, w_y)
        
        t_px = px + sigma * gx
        t_py = py + sigma * gy
        norm = torch.sqrt(t_px**2 + t_py**2 + 1e-12)
        denom = torch.clamp(norm, min=1.0)
        px = t_px / denom
        py = t_py / denom
        
        pad_px = nn.functional.pad(px, (0, K - 1, 0, 0), mode="constant", value=0.0)
        pad_py = nn.functional.pad(py, (0, 0, 0, K - 1), mode="constant", value=0.0)
        div = nn.functional.conv2d(pad_px, w_x_adj) + nn.functional.conv2d(pad_py, w_y_adj)
        
        v_tilde = v - tau * div
        v_old = v.clone()
        for _ in range(3):
            exp_neg = torch.exp(-v)
            f_val = v - v_tilde + tau * lam_map * (1.0 - I_clamped * exp_neg)
            f_prime = 1.0 + tau * lam_map * I_clamped * exp_neg
            v = v - f_val / f_prime
        v_bar = 2.0 * v - v_old
        
    out = torch.exp(v)
    out[cfar_mask] = I_clamped[cfar_mask]
    out_np = out.squeeze().cpu().numpy()
    return out_np * (im.mean() / out_np.mean())


# --- МЕТОД 10: Нейросетевой DIP (Deep Image Prior с гамма-правдоподобием) ---
def filter_rff_dip(im: np.ndarray, n_iter: int = 50) -> np.ndarray:
    H, W = im.shape
    y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing="ij")
    coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)
    
    # RFF mapping (16 random frequencies)
    B_mat = torch.randn(16, 2, device=device) * 8.0
    proj = 2.0 * math.pi * torch.einsum("nc,bchw->bnhw", B_mat, coords)
    z = torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)
    
    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = nn.Sequential(nn.Conv2d(32, 24, 3, padding=1), nn.LeakyReLU(0.2))
            self.e2 = nn.Sequential(nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
            self.d1 = nn.Sequential(nn.Conv2d(48 + 24, 24, 3, padding=1), nn.LeakyReLU(0.2))
            self.out = nn.Conv2d(24, 1, 1)
        def forward(self, x):
            f1 = self.e1(x)
            f2 = self.e2(f1)
            up = nn.functional.interpolate(f2, size=f1.shape[2:], mode="bilinear", align_corners=False)
            cat = self.d1(torch.cat([up, f1], dim=1))
            return self.out(cat)
            
    net = SmallUNet().to(device)
    opt = optim.Adam(net.parameters(), lr=2e-3)
    I_clamped = torch.clamp(torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(device), min=1e-6)
    
    for _ in range(n_iter):
        opt.zero_grad()
        log_pred = net(z)
        pred_R = torch.exp(log_pred)
        loss = (log_pred + I_clamped / pred_R).mean()
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        out = torch.exp(net(z)).squeeze().cpu().numpy()
    return out * (im.mean() / out.mean())


# =========================================================================
# 4. ПОЛНЫЙ ЭКСПЕРИМЕНТАЛЬНЫЙ ПЕРЕБОР И СРАВНЕНИЕ ВСЕХ МЕТОДОВ
# =========================================================================

def run_complete_benchmark():
    print("========================================================================")
    print("ЗАПУСК ПОЛНОГО ЭКСПЕРИМЕНТАЛЬНОГО СРАВНЕНИЯ МЕТОДОВ РАСШУМЛЕНИЯ SAR")
    print("========================================================================")
    
    gt, noisy, homo_mask = generate_synthetic_sar_benchmark(size=512, looks=4.4, seed=42)
    print(f"Сгенерирован синтетический полигон 512x512 с L=4.4 эквивалентных отсчетов Sentinel-1.")
    print(f"Целевая дисперсия спекла 1/L = {1.0/4.4:.4f}")
    
    # Загрузка обученных нейросетевых моделей
    from train_sar_models import SAR_ResNet, SAR_MERLIN_BlindSpot, SAR_Restormer
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, "HTML Визуализация", "веса")
    
    m_resnet = SAR_ResNet(in_channels=1, num_features=64, num_layers=17).to(device)
    m_resnet.load_state_dict(torch.load(os.path.join(weights_dir, "sar_resnet_vitale.pth"), map_location=device))
    m_resnet.eval()
    
    m_merlin = SAR_MERLIN_BlindSpot(in_channels=1, num_features=48).to(device)
    m_merlin.load_state_dict(torch.load(os.path.join(weights_dir, "sar_merlin_blindspot.pth"), map_location=device))
    m_merlin.eval()
    
    m_restormer = SAR_Restormer(in_channels=1, base_dim=32).to(device)
    m_restormer.load_state_dict(torch.load(os.path.join(weights_dir, "sar_restormer.pth"), map_location=device))
    m_restormer.eval()
    
    def filter_dl_model(im: np.ndarray, model) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(im.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                out = model(t)
            res = out.squeeze().cpu().numpy()
        return res * (im.mean() / max(res.mean(), 1e-8))

    # Словарь исследуемых методов
    methods = {
        "0. Исходный (Noisy)": lambda x: x,
        "1. Lee Filter (3x3)": lambda x: filter_lee(x, w=3, looks=4.4),
        "2. Lee Filter (5x5)": lambda x: filter_lee(x, w=5, looks=4.4),
        "3. Lee Filter (7x7)": lambda x: filter_lee(x, w=7, looks=4.4),
        "4. Frost Filter (3x3, a=1.0)": lambda x: filter_frost(x, w=3, a=1.0),
        "5. Frost Filter (5x5, a=2.0)": lambda x: filter_frost(x, w=5, a=2.0),
        "6. Kuan Filter (5x5)": lambda x: filter_kuan(x, w=5, looks=4.4),
        "7. Gamma MAP (3x3)": lambda x: filter_gamma_map(x, w=3, looks=4.4),
        "8. Gamma MAP (5x5)": lambda x: filter_gamma_map(x, w=5, looks=4.4),
        "9. SAR-NLM (Non-Local Means)": lambda x: filter_sar_nlm(x, patch_rad=2, search_rad=5, h=0.35),
        "10. Wavelet BayesShrink (Log)": lambda x: filter_wavelet_sar(x, threshold_scale=1.0),
        "11. Total Variation ROF (lam=1.5)": lambda x: filter_tv_rof(x, lam=1.5, n_iter=25),
        "12. Total Variation ROF (lam=4.0)": lambda x: filter_tv_rof(x, lam=4.0, n_iter=25),
        "13. Fractional TV (alpha=1.25, lam=1.5)": lambda x: filter_ftv(x, alpha=1.25, lam=1.5, n_iter=25),
        "14. Fractional TV (alpha=1.25, lam=4.0)": lambda x: filter_ftv(x, alpha=1.25, lam=4.0, n_iter=25),
        "15. Fractional TV (alpha=1.25, lam=8.0)": lambda x: filter_ftv(x, alpha=1.25, lam=8.0, n_iter=25),
        "16. Joint FTV-SARP (lam=3.5..11.5)": lambda x: filter_ftv_sarp(x, lam_base=3.5, lam_boost=8.0, n_iter=25),
        "17. Joint FTV-SARP (lam=5.0..15.0)": lambda x: filter_ftv_sarp(x, lam_base=5.0, lam_boost=10.0, n_iter=25),
        "18. RFF-DIP (Gamma NLL, 50 iters)": lambda x: filter_rff_dip(x, n_iter=50),
        "19. SAR-ResNet (Vitale et al.)": lambda x: filter_dl_model(x, m_resnet),
        "20. MERLIN Blind-Spot CNN": lambda x: filter_dl_model(x, m_merlin),
        "21. SAR-Restormer (Transformer)": lambda x: filter_dl_model(x, m_restormer),
    }
    
    results = []
    filtered_images = {}
    
    for name, func in methods.items():
        t0 = time.time()
        filt_img = func(noisy)
        elapsed_ms = (time.time() - t0) * 1000.0
        
        m = compute_benchmark_metrics(gt, noisy, filt_img, homo_mask, looks=4.4)
        m["name"] = name
        m["time_ms"] = round(elapsed_ms, 1)
        results.append(m)
        filtered_images[name] = filt_img
        print(f"[{name:<35}] PSNR: {m['psnr']:5.2f} dB | SSIM: {m['ssim']:6.4f} | EPI: {m['epi_gt']:6.4f} | ENL: {m['enl_homo']:6.2f} | Var(r): {m['ratio_var']:6.4f} | Time: {m['time_ms']:6.1f} ms")
        
    # Сохранение результатов в JSON и CSV
    out_dir = os.path.join(base_dir, "HTML Визуализация", "результаты_фильтрации", "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "benchmark_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "benchmark_metrics.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[УСПЕХ] Метрики сохранены в {out_dir}\\benchmark_metrics.csv")
    
    # Генерация наглядной сводной панели изображений (12 ключевых методов, сетка 2 x 6)
    selected_vis = [
        ("Ground Truth", gt),
        ("0. Noisy (L=4.4)", noisy),
        ("1. Lee 5x5", filtered_images["2. Lee Filter (5x5)"]),
        ("2. Frost 5x5", filtered_images["5. Frost Filter (5x5, a=2.0)"]),
        ("3. Gamma MAP 5x5", filtered_images["8. Gamma MAP (5x5)"]),
        ("4. SAR-NLM", filtered_images["9. SAR-NLM (Non-Local Means)"]),
        ("5. Wavelet Bayes", filtered_images["10. Wavelet BayesShrink (Log)"]),
        ("6. TV ROF (lam=1.5)", filtered_images["11. Total Variation ROF (lam=1.5)"]),
        ("7. Joint FTV-SARP", filtered_images["16. Joint FTV-SARP (lam=3.5..11.5)"]),
        ("8. SAR-ResNet (Vitale)", filtered_images["19. SAR-ResNet (Vitale et al.)"]),
        ("9. MERLIN Blind-Spot", filtered_images["20. MERLIN Blind-Spot CNN"]),
        ("10. SAR-Restormer", filtered_images["21. SAR-Restormer (Transformer)"]),
    ]
    
    vis_tiles = []
    for title, arr in selected_vis:
        arr_db = 10.0 * np.log10(np.maximum(arr, 1e-4))
        norm = np.clip((arr_db - (-20.0)) / (5.0 - (-20.0)), 0.0, 1.0)
        img_u8 = (norm * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8).convert("RGB")
        vis_tiles.append(pil_img)
        
    # Собираем коллаж 2 x 6
    tile_w, tile_h = 512, 512
    collage = Image.new("RGB", (tile_w * 6, tile_h * 2))
    for i, t_img in enumerate(vis_tiles):
        row = i // 6
        col = i % 6
        collage.paste(t_img, (col * tile_w, row * tile_h))
        
    collage_path = os.path.join(out_dir, "benchmark_collage.png")
    collage.save(collage_path)
    print(f"[УСПЕХ] Сводный визуальный коллаж сохранен: {collage_path}")
    
    return df

if __name__ == "__main__":
    run_complete_benchmark()
