# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('pipeline_search_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

# Combine 1, 2, and 3 step
all_pipes = []
for p in d['results_1step']:
    p['steps'] = 1
    all_pipes.append(p)
for p in d['results_2step']:
    p['steps'] = 2
    all_pipes.append(p)
for p in d['results_3step']:
    p['steps'] = 3
    all_pipes.append(p)

print(f"Total pipelines evaluated: {len(all_pipes)}")

# Pareto frontier in (PSNR, SSIM, EPI, ENL)
# For each steps count, find top Pareto points
for steps in [1, 2, 3]:
    subset = [p for p in all_pipes if p['steps'] == steps]
    # Rank by PSNR
    top_psnr = sorted(subset, key=lambda x: x['psnr'], reverse=True)[:5]
    # Rank by SSIM
    top_ssim = sorted(subset, key=lambda x: x['ssim'], reverse=True)[:5]
    # Rank by EPI (with ENL > 3.5)
    top_epi = sorted([p for p in subset if p['enl'] >= 3.5], key=lambda x: x['epi'], reverse=True)[:5]
    # Rank by ENL (with EPI > 0.5)
    top_enl_struct = sorted([p for p in subset if p['epi'] >= 0.5], key=lambda x: x['enl'], reverse=True)[:5]
    
    print(f"\n=================== {steps}-STEP PIPELINES ===================")
    print("--- Top by PSNR (дБ) ---")
    for x in top_psnr:
        print(f"  {x['pipeline']:28s} | PSNR: {x['psnr']:5.2f} дБ | SSIM: {x['ssim']:6.4f} | ENL: {x['enl']:5.2f} | EPI: {x['epi']:6.4f} | Bias: {x['bias_db']:6.3f} дБ")
    print("--- Top by SSIM ---")
    for x in top_ssim:
        print(f"  {x['pipeline']:28s} | SSIM: {x['ssim']:6.4f} | PSNR: {x['psnr']:5.2f} дБ | ENL: {x['enl']:5.2f} | EPI: {x['epi']:6.4f} | Bias: {x['bias_db']:6.3f} дБ")
    print("--- Top by EPI (при сохранении ENL >= 3.5) ---")
    for x in top_epi:
        print(f"  {x['pipeline']:28s} | EPI: {x['epi']:6.4f} | ENL: {x['enl']:5.2f} | PSNR: {x['psnr']:5.2f} дБ | SSIM: {x['ssim']:6.4f} | Bias: {x['bias_db']:6.3f} дБ")
    print("--- Top by ENL (при сохранении границ EPI >= 0.5) ---")
    for x in top_enl_struct:
        print(f"  {x['pipeline']:28s} | ENL: {x['enl']:5.2f} | EPI: {x['epi']:6.4f} | PSNR: {x['psnr']:5.2f} дБ | SSIM: {x['ssim']:6.4f} | Bias: {x['bias_db']:6.3f} дБ")
