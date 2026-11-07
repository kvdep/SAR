import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "html", "index.html"))

if not os.path.exists(HTML_PATH):
    raise FileNotFoundError(f"Файл визуализации не найден: {HTML_PATH}")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    text = f.read()

prefix = "const DATA = "
start = text.find(prefix) + len(prefix)
end = text.find(";\n", start)
data = json.loads(text[start:end])

print("Scenes:", [k for k in data.keys() if not k.startswith("_")])
benchmarks = data.get("_benchmarks", [])
print("Benchmarks count:", len(benchmarks))
for b in benchmarks:
    bid = b["id"]
    pairs = b["pairs"]
    print(f"\n--- Benchmark: {bid} (Total {len(pairs)} pairs) ---")
    for k, p in pairs.items():
        m = p.get("metrics", {})
        enl = m.get("enl")
        epi = m.get("epi")
        bias = m.get("bias_db")
        psnr = m.get("psnr")
        ssim = m.get("ssim")
        print(f"  {k:15s} | ENL: {str(enl):5s} | EPI: {str(epi):5s} | Bias: {str(bias):6s} dB | PSNR: {str(psnr):5s} | SSIM: {str(ssim):5s} | {p['name']}")

for sc_key in ['forest', 'city', 'water', 'desert', 'mountain', 'benevento', 'rome', 'serre_poncon']:
    sc = data[sc_key]
    pipes = sc['pipelines']
    print(f"\n--- SCENE: {sc_key} ({len(pipes)} pipelines) ---")
    for pk in ['raw', 'boxcar', 'median', 'lee', 'gamma', 'ftv', 'resnet', 'merlin', 'restormer', 'multitemp_gt']:
        if pk in pipes:
            m = pipes[pk]['metrics']
            print(f"  {pk:12s} | ENL: {str(m.get('enl')):6s} | EPI: {str(m.get('epi')):6s} | Bias: {str(m.get('bias_db')):7s} dB | PSNR: {str(m.get('psnr')):5s} | SSIM: {str(m.get('ssim')):5s}")

