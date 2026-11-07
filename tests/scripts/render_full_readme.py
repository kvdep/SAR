import re
import markdown
from seleniumbase import SB

readme_path = r"D:\ПРОФ\ВУЗ\ВТСК\ОМЕЛА GIT\src\parsers\sentinel1C_parser\README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    md_text = f.read()

html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.0/github-markdown.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ],
        throwOnError: false
    });
});
</script>
<style>
.markdown-body {
    box-sizing: border-box;
    min-width: 200px;
    max-width: 1000px;
    margin: 0 auto;
    padding: 40px;
}
</style>
</head>
<body class="markdown-body">
CONTENT_PLACEHOLDER
</body>
</html>
"""

def protect_math(text):
    math_tokens = []
    def repl(m):
        math_tokens.append(m.group(0))
        return f"MATHPROTECTED{len(math_tokens)-1}TOKEN"
    text = re.sub(r"\$[^$\n]+\$", repl, text)
    html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])
    for idx, token in enumerate(math_tokens):
        html = html.replace(f"MATHPROTECTED{idx}TOKEN", token)
    return html

html_body = protect_math(md_text)
full_html = html_template.replace("CONTENT_PLACEHOLDER", html_body)

out_html = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\full_readme_rendered.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(full_html)

out_png = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\full_readme_rendered.png"
with SB(uc=False, headless=True) as sb:
    sb.open("file:///" + out_html.replace("\\", "/"))
    sb.sleep(2)
    sb.set_window_size(1200, 3200)
    sb.sleep(1)
    sb.save_screenshot(out_png)

print("Saved full_readme_rendered.png")
