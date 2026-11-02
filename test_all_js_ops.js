// test_all_js_ops.js
const W = 256;
const H = 256;
const raw = new Float32Array(W * H);
for (let i = 0; i < W * H; i++) raw[i] = Math.random();

function getPixel(arr, w, h, x, y) {
    const cx = Math.max(0, Math.min(w - 1, x));
    const cy = Math.max(0, Math.min(h - 1, y));
    return arr[cy * w + cx];
}

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
            out[y * w + x] = mu + weight * (arr[y * w + x] - mu);
        }
    }
    return out;
}

function filterEnhLee(arr, w, h, ksize = 5, looks = 4.4, k = 1.0) {
    const out = new Float32Array(w * h);
    const rad = Math.floor(ksize / 2);
    const count = ksize * ksize;
    const cu = 1.0 / Math.sqrt(looks);
    const cmax = Math.sqrt(1.0 + 2.0 / looks);
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
                const weight = Math.exp(-k * (ci - cu) / (cmax - ci + 1e-8));
                out[y * w + x] = mu + weight * (center - mu);
            } else {
                out[y * w + x] = center;
            }
        }
    }
    return out;
}

function filterKuan(arr, w, h, ksize = 5, looks = 4.4) {
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
            const weight = Math.max(0, Math.min(1.0, (1.0 - cu_sq / (ci_sq + 1e-8)) / (1.0 + cu_sq)));
            out[y * w + x] = mu + weight * (arr[y * w + x] - mu);
        }
    }
    return out;
}

function filterFrost(arr, w, h, ksize = 5, damping = 2.0) {
    const out = new Float32Array(w * h);
    const rad = Math.floor(ksize / 2);
    const count = ksize * ksize;
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
            let sumW = 0, sumVal = 0;
            for (let dy = -rad; dy <= rad; dy++) {
                for (let dx = -rad; dx <= rad; dx++) {
                    const d = Math.sqrt(dx * dx + dy * dy);
                    const w_k = Math.exp(-damping * ci * d);
                    const v = getPixel(arr, w, h, x + dx, y + dy);
                    sumVal += v * w_k;
                    sumW += w_k;
                }
            }
            out[y * w + x] = sumVal / (sumW + 1e-8);
        }
    }
    return out;
}

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

console.log("Testing all 8 classical JS filters on 256x256...");
const ops = [
    ["Boxcar", () => filterBoxcar(raw, W, H, 5)],
    ["Median", () => filterMedian(raw, W, H, 5)],
    ["Lee", () => filterLee(raw, W, H, 5, 4.4)],
    ["EnhLee", () => filterEnhLee(raw, W, H, 5, 4.4)],
    ["Kuan", () => filterKuan(raw, W, H, 5, 4.4)],
    ["Frost", () => filterFrost(raw, W, H, 5, 2.0)],
    ["Bilateral", () => filterBilateral(raw, W, H, 5, 0.15, 2.5)],
    ["GammaMap", () => filterGammaMap(raw, W, H, 5, 4.4)]
];

for (const [name, fn] of ops) {
    const t0 = performance.now();
    const res = fn();
    console.log(`  ${name.padEnd(12)}: ${(performance.now() - t0).toFixed(2)} ms (mean: ${(res.reduce((a,b)=>a+b,0)/res.length).toFixed(4)})`);
}
