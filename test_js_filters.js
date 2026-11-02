// test_js_filters.js
// Verification of client-side JavaScript SAR filtering engine

const W = 256;
const H = 256;
const raw = new Float32Array(W * H);
for (let i = 0; i < W * H; i++) {
    raw[i] = Math.random();
}

function getPixel(arr, w, h, x, y) {
    const cx = Math.max(0, Math.min(w - 1, x));
    const cy = Math.max(0, Math.min(h - 1, y));
    return arr[cy * w + cx];
}

// 1. Boxcar / Uniform Mean
function filterBoxcar(arr, w, h, ksize = 5) {
    const out = new Float32Array(w * h);
    const rad = Math.floor(ksize / 2);
    const count = ksize * ksize;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let sum = 0;
            for (let dy = -rad; dy <= rad; dy++) {
                for (let dx = -rad; dx <= rad; dx++) {
                    sum += getPixel(arr, w, h, x + dx, y + dy);
                }
            }
            out[y * w + x] = sum / count;
        }
    }
    return out;
}

// 2. Median Filter
function filterMedian(arr, w, h, ksize = 5) {
    const out = new Float32Array(w * h);
    const rad = Math.floor(ksize / 2);
    const count = ksize * ksize;
    const mid = Math.floor(count / 2);
    const buf = new Float32Array(count);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let idx = 0;
            for (let dy = -rad; dy <= rad; dy++) {
                for (let dx = -rad; dx <= rad; dx++) {
                    buf[idx++] = getPixel(arr, w, h, x + dx, y + dy);
                }
            }
            buf.sort();
            out[y * w + x] = buf[mid];
        }
    }
    return out;
}

// 3. Lee Filter
function filterLee(arr, w, h, ksize = 5, looks = 4.4) {
    const out = new Float32Array(w * h);
    const rad = Math.floor(ksize / 2);
    const count = ksize * ksize;
    const cu_sq = 1.0 / looks;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let sum = 0, sum_sq = 0;
            for (let dy = -rad; dy <= rad; dy++) {
                for (let dx = -rad; dx <= rad; dx++) {
                    const v = getPixel(arr, w, h, x + dx, y + dy);
                    sum += v;
                    sum_sq += v * v;
                }
            }
            const mu = sum / count;
            const variance = Math.max(0, sum_sq / count - mu * mu);
            const ci_sq = variance / (mu * mu + 1e-8);
            const weight = Math.max(0, Math.min(1.0, 1.0 - cu_sq / (ci_sq + 1e-8)));
            const center = arr[y * w + x];
            out[y * w + x] = mu + weight * (center - mu);
        }
    }
    return out;
}

// 4. Bilateral Filter
function filterBilateral(arr, w, h, d = 5, sigmaColor = 0.15, sigmaSpace = 2.5) {
    const out = new Float32Array(w * h);
    const rad = Math.floor(d / 2);
    const colorCoeff = -0.5 / (sigmaColor * sigmaColor + 1e-8);
    const spaceCoeff = -0.5 / (sigmaSpace * sigmaSpace + 1e-8);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const centerVal = arr[y * w + x];
            let sumWeight = 0, sumVal = 0;
            for (let dy = -rad; dy <= rad; dy++) {
                for (let dx = -rad; dx <= rad; dx++) {
                    const val = getPixel(arr, w, h, x + dx, y + dy);
                    const dColor = val - centerVal;
                    const dDistSq = dx * dx + dy * dy;
                    const wSpatial = Math.exp(dDistSq * spaceCoeff);
                    const wRange = Math.exp(dColor * dColor * colorCoeff);
                    const weight = wSpatial * wRange;
                    sumWeight += weight;
                    sumVal += val * weight;
                }
            }
            out[y * w + x] = sumVal / (sumWeight + 1e-8);
        }
    }
    return out;
}

// 5. Gamma-MAP Filter
function filterGammaMap(arr, w, h, ksize = 5, looks = 4.4) {
    const out = new Float32Array(w * h);
    const rad = Math.floor(ksize / 2);
    const count = ksize * ksize;
    const cu = 1.0 / Math.sqrt(looks);
    const cmax = Math.sqrt(2.0) * cu;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let sum = 0, sum_sq = 0;
            for (let dy = -rad; dy <= rad; dy++) {
                for (let dx = -rad; dx <= rad; dx++) {
                    const v = getPixel(arr, w, h, x + dx, y + dy);
                    sum += v;
                    sum_sq += v * v;
                }
            }
            const mu = sum / count;
            const variance = Math.max(0, sum_sq / count - mu * mu);
            const ci = Math.sqrt(variance) / (mu + 1e-8);
            const center = arr[y * w + x];
            if (ci <= cu) {
                out[y * w + x] = mu;
            } else if (ci < cmax) {
                const a = (1.0 + cu * cu) / (ci * ci - cu * cu + 1e-8);
                const b = (a - looks - 1.0) * mu;
                const d = b * b + 4.0 * a * looks * center * mu;
                out[y * w + x] = (b + Math.sqrt(Math.max(0, d))) / (2.0 * a);
            } else {
                out[y * w + x] = center;
            }
        }
    }
    return out;
}

console.log("Testing filter execution times...");
const t0 = performance.now();
const b = filterBoxcar(raw, W, H, 5);
console.log(`Boxcar 5x5: ${(performance.now() - t0).toFixed(2)} ms`);

const t1 = performance.now();
const m = filterMedian(raw, W, H, 5);
console.log(`Median 5x5: ${(performance.now() - t1).toFixed(2)} ms`);

const t2 = performance.now();
const l = filterLee(raw, W, H, 5, 4.4);
console.log(`Lee 5x5: ${(performance.now() - t2).toFixed(2)} ms`);

const t3 = performance.now();
const bil = filterBilateral(raw, W, H, 5, 0.15, 2.5);
console.log(`Bilateral: ${(performance.now() - t3).toFixed(2)} ms`);

const t4 = performance.now();
const g = filterGammaMap(raw, W, H, 5, 4.4);
console.log(`Gamma MAP: ${(performance.now() - t4).toFixed(2)} ms`);

// Metric computation in pure JS
function computeMetrics(raw, filt, gt, w, h) {
    let sumR = 0, sumF = 0;
    const n = w * h;
    for (let i = 0; i < n; i++) {
        sumR += raw[i];
        sumF += filt[i];
    }
    const muR = sumR / n;
    const muF = sumF / n;
    const biasDb = 10.0 * Math.log10(Math.max(1e-8, muF) / Math.max(1e-8, muR));

    // Homogeneous mask for ENL
    let sumHomo = 0, sumHomoSq = 0, countHomo = 0;
    const rad = 3;
    const count = 49;
    const cu = 1.0 / Math.sqrt(4.4);
    for (let y = rad; y < h - rad; y += 2) {
        for (let x = rad; x < w - rad; x += 2) {
            let s = 0, sq = 0;
            for (let dy = -rad; dy <= rad; dy++) {
                for (let dx = -rad; dx <= rad; dx++) {
                    const v = filt[(y + dy) * w + (x + dx)];
                    s += v;
                    sq += v * v;
                }
            }
            const m = s / count;
            const variance = Math.max(0, sq / count - m * m);
            const ci = Math.sqrt(variance) / (m + 1e-8);
            if (ci < 1.15 * cu) {
                const cVal = filt[y * w + x];
                sumHomo += cVal;
                sumHomoSq += cVal * cVal;
                countHomo++;
            }
        }
    }
    let enl = 0;
    if (countHomo > 20) {
        const mH = sumHomo / countHomo;
        const vH = Math.max(1e-8, sumHomoSq / countHomo - mH * mH);
        enl = (mH * mH) / vH;
    } else {
        let vAll = 0;
        for (let i = 0; i < n; i++) vAll += (filt[i] - muF) * (filt[i] - muF);
        enl = (muF * muF) / (vAll / n + 1e-8);
    }

    // EPI via discrete Laplacian
    let sumLR = 0, sumLF = 0, sumLR2 = 0, sumLF2 = 0, sumLRF = 0;
    let lapCount = 0;
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            const idx = y * w + x;
            const lR = 4 * raw[idx] - raw[idx - 1] - raw[idx + 1] - raw[idx - w] - raw[idx + w];
            const lF = 4 * filt[idx] - filt[idx - 1] - filt[idx + 1] - filt[idx - w] - filt[idx + w];
            sumLR += lR;
            sumLF += lF;
            sumLR2 += lR * lR;
            sumLF2 += lF * lF;
            sumLRF += lR * lF;
            lapCount++;
        }
    }
    const muLR = sumLR / lapCount;
    const muLF = sumLF / lapCount;
    const varLR = sumLR2 / lapCount - muLR * muLR;
    const varLF = sumLF2 / lapCount - muLF * muLF;
    const covLRF = sumLRF / lapCount - muLR * muLF;
    const epi = (varLR > 1e-8 && varLF > 1e-8) ? covLRF / Math.sqrt(varLR * varLF) : 1.0;

    const res = {
        enl: parseFloat(enl.toFixed(2)),
        epi: parseFloat(epi.toFixed(3)),
        bias_db: parseFloat(biasDb.toFixed(3)),
    };

    if (gt) {
        let mse = 0, minGt = 1e9, maxGt = -1e9;
        let sumG = 0, sumG2 = 0, sumGF = 0;
        for (let i = 0; i < n; i++) {
            const diff = gt[i] - filt[i];
            mse += diff * diff;
            if (gt[i] < minGt) minGt = gt[i];
            if (gt[i] > maxGt) maxGt = gt[i];
            sumG += gt[i];
            sumG2 += gt[i] * gt[i];
            sumGF += gt[i] * filt[i];
        }
        mse /= n;
        const dr = maxGt - minGt;
        const psnr = mse > 0 ? 10.0 * Math.log10((dr * dr) / mse) : 99.9;

        const muG = sumG / n;
        const varG = sumG2 / n - muG * muG;
        const varF = sumLF2 / lapCount; // or recompute
        let vF_full = 0;
        for (let i = 0; i < n; i++) vF_full += (filt[i] - muF) * (filt[i] - muF);
        vF_full /= n;
        const covGF = sumGF / n - muG * muF;
        const c1 = (0.01 * dr) * (0.01 * dr);
        const c2 = (0.03 * dr) * (0.03 * dr);
        const ssim = ((2 * muG * muF + c1) * (2 * covGF + c2)) / ((muG * muG + muF * muF + c1) * (varG + vF_full + c2));

        res.psnr = parseFloat(psnr.toFixed(2));
        res.ssim = parseFloat(ssim.toFixed(4));
    }

    return res;
}

const tM = performance.now();
const metrics = computeMetrics(raw, b, raw, W, H);
console.log(`Metrics computed in ${(performance.now() - tM).toFixed(2)} ms:`, metrics);

