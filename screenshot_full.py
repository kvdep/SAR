from seleniumbase import SB

out_html = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\rendered_readme.html"
out_png = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\rendered_readme_full.png"

with SB(uc=False, headless=True) as sb:
    sb.open("file:///" + out_html.replace("\\", "/"))
    sb.sleep(2)
    # Set window size to capture the whole page
    sb.set_window_size(1200, 2400)
    sb.sleep(1)
    sb.save_screenshot(out_png)

print("Saved full screenshot")
