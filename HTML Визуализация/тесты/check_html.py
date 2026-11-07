import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
p = os.path.join(SCRIPT_DIR, "..", "html", "index.html")

if os.path.exists(p):
    mtime = os.path.getmtime(p)
    print(f"File size: {os.path.getsize(p) / 1024 / 1024:.2f} MB")
    print(f"Last modified: {time.ctime(mtime)}")

    with open(p, "r", encoding="utf-8") as f:
        full = f.read()

    print("open-benchmarks-btn in full:", "open-benchmarks-btn" in full)
    print("benchmarks-modal in full:", "benchmarks-modal" in full)
    print("_benchmarks in full:", "_benchmarks" in full)
    print("value='resnet' in full:", 'value="resnet"' in full)
    print("value='merlin' in full:", 'value="merlin"' in full)
    print("value='restormer' in full:", 'value="restormer"' in full)
else:
    print(f"File not found: {p}")

print("\nAll HTML files in SAR root directory (must be empty of duplicate bundles):")
for f in os.listdir(PROJECT_ROOT):
    if f.endswith(".html"):
        fp = os.path.join(PROJECT_ROOT, f)
        print(f"  {f:40s} | {os.path.getsize(fp)/1024/1024:.2f} MB | {time.ctime(os.path.getmtime(fp))}")
