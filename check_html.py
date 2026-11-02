import os
import time

p = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\sar_advanced_comparison.html"
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

# Check if there are other html files in D:\ПРОФ\ВУЗ\ВТСК\SAR
print("\nAll HTML files in SAR directory:")
for f in os.listdir(r"D:\ПРОФ\ВУЗ\ВТСК\SAR"):
    if f.endswith(".html"):
        fp = os.path.join(r"D:\ПРОФ\ВУЗ\ВТСК\SAR", f)
        print(f"  {f:40s} | {os.path.getsize(fp)/1024/1024:.2f} MB | {time.ctime(os.path.getmtime(fp))}")
