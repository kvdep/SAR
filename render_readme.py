import re
from seleniumbase import SB

readme_path = r"D:\ПРОФ\ВУЗ\ВТСК\ОМЕЛА GIT\src\parsers\sentinel1C_parser\README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    md_text = f.read()

# We want to render this markdown into HTML with KaTeX and GitHub Markdown style
# Let's inspect how math is written and wrap into HTML.
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
            {left: '$', right: '$', display: false},
            {left: '\\\\(', right: '\\\\)', display: false},
            {left: '\\\\[', right: '\\\\]', display: true}
        ],
        throwOnError: false
    });
});
</script>
<style>
.markdown-body {
    box-sizing: border-box;
    min-width: 200px;
    max-width: 980px;
    margin: 0 auto;
    padding: 45px;
}
</style>
</head>
<body class="markdown-body">
CONTENT_PLACEHOLDER
</body>
</html>
"""

# Let's use markdown library or simple conversion
import markdown

html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
full_html = html_template.replace("CONTENT_PLACEHOLDER", html_body)

out_html = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\rendered_readme.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(full_html)

out_png = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\rendered_readme.png"
with SB(uc=False, headless=True) as sb:
    sb.open("file:///" + out_html.replace("\\", "/"))
    sb.sleep(3)
    sb.save_screenshot(out_png)

print("Rendered successfully to", out_png)
