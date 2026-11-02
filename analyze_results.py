# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('pipeline_search_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

out_f = open('results_summary.txt', 'w', encoding='utf-8')
def print_out(*args):
    s = ' '.join(str(a) for a in args)
    out_f.write(s + '\n')
    print(s)

print_out('Raw input metrics:', data['raw'])

print_out('\n=== ALL 12 1-STEP PIPELINES (by Composite Score) ===')
for i, item in enumerate(data['results_1step'], 1):
    print_out(f"{i:2d}. {item['pipeline']:10s} | Score: {item['score']:5.2f} | PSNR: {item['psnr']:5.2f} дБ | SSIM: {item['ssim']:6.4f} | ENL: {item['enl']:5.2f} | EPI: {item['epi']:6.4f} | Bias: {item['bias_db']:6.3f} дБ")

print_out('\n=== TOP-15 2-STEP PIPELINES (by Composite Score) ===')
for i, item in enumerate(data['results_2step'][:15], 1):
    print_out(f"{i:2d}. {item['pipeline']:20s} | Score: {item['score']:5.2f} | PSNR: {item['psnr']:5.2f} дБ | SSIM: {item['ssim']:6.4f} | ENL: {item['enl']:5.2f} | EPI: {item['epi']:6.4f} | Bias: {item['bias_db']:6.3f} дБ")

print_out('\n=== TOP-15 2-STEP PIPELINES (by PSNR) ===')
by_psnr_2 = sorted(data['results_2step'], key=lambda x: x['psnr'], reverse=True)
for i, item in enumerate(by_psnr_2[:15], 1):
    print_out(f"{i:2d}. {item['pipeline']:20s} | PSNR: {item['psnr']:5.2f} дБ | SSIM: {item['ssim']:6.4f} | ENL: {item['enl']:5.2f} | EPI: {item['epi']:6.4f} | Bias: {item['bias_db']:6.3f} дБ | Score: {item['score']:5.2f}")

print_out('\n=== TOP-15 3-STEP PIPELINES (by Composite Score) ===')
for i, item in enumerate(data['results_3step'][:15], 1):
    print_out(f"{i:2d}. {item['pipeline']:28s} | Score: {item['score']:5.2f} | PSNR: {item['psnr']:5.2f} дБ | SSIM: {item['ssim']:6.4f} | ENL: {item['enl']:5.2f} | EPI: {item['epi']:6.4f} | Bias: {item['bias_db']:6.3f} дБ")

print_out('\n=== TOP-15 3-STEP PIPELINES (by PSNR) ===')
by_psnr_3 = sorted(data['results_3step'], key=lambda x: x['psnr'], reverse=True)
for i, item in enumerate(by_psnr_3[:15], 1):
    print_out(f"{i:2d}. {item['pipeline']:28s} | PSNR: {item['psnr']:5.2f} дБ | SSIM: {item['ssim']:6.4f} | ENL: {item['enl']:5.2f} | EPI: {item['epi']:6.4f} | Bias: {item['bias_db']:6.3f} дБ | Score: {item['score']:5.2f}")

print_out('\n=== TOP-10 3-STEP PIPELINES (by SSIM) ===')
by_ssim_3 = sorted(data['results_3step'], key=lambda x: x['ssim'], reverse=True)
for i, item in enumerate(by_ssim_3[:10], 1):
    print_out(f"{i:2d}. {item['pipeline']:28s} | SSIM: {item['ssim']:6.4f} | PSNR: {item['psnr']:5.2f} дБ | ENL: {item['enl']:5.2f} | EPI: {item['epi']:6.4f} | Bias: {item['bias_db']:6.3f} дБ | Score: {item['score']:5.2f}")

print_out('\n=== TOP-10 3-STEP PIPELINES (by ENL - Максимальное подавление спекла) ===')
by_enl_3 = sorted(data['results_3step'], key=lambda x: x['enl'], reverse=True)
for i, item in enumerate(by_enl_3[:10], 1):
    print_out(f"{i:2d}. {item['pipeline']:28s} | ENL: {item['enl']:5.2f} | EPI: {item['epi']:6.4f} | PSNR: {item['psnr']:5.2f} дБ | SSIM: {item['ssim']:6.4f} | Bias: {item['bias_db']:6.3f} дБ | Score: {item['score']:5.2f}")

print_out('\n=== ХУДШИЕ 5 3-STEP PIPELINES (по Composite Score) ===')
for i, item in enumerate(data['results_3step'][-5:], 1):
    print_out(f"{i:2d}. {item['pipeline']:28s} | Score: {item['score']:5.2f} | PSNR: {item['psnr']:5.2f} дБ | SSIM: {item['ssim']:6.4f} | ENL: {item['enl']:5.2f} | EPI: {item['epi']:6.4f} | Bias: {item['bias_db']:6.3f} дБ")

# Validation on forest
with open('forest_top_pipelines.json', 'r', encoding='utf-8') as f:
    f_data = json.load(f)

print_out('\n=== ВАЛИДАЦИЯ НА РЕАЛЬНОМ ЛЕСЕ (ТОП КАНДИДАТЫ) ===')
for i, item in enumerate(f_data, 1):
    print_out(f"{i:2d}. [{item['steps']} шага] {item['pipeline']:28s} | ENL: {item['enl']:6.2f} | EPI: {item['epi']:6.4f} | Bias: {item['bias_db']:6.3f} дБ | Var(r): {item['var_r']:6.4f}")
