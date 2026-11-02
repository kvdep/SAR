import re
import markdown
from seleniumbase import SB

test_md = """
### 4. Этапы конвейера обработки (Pipeline)

1. **Парсинг входного GeoJSON:**
    * Извлечение координат области интереса (Polygon или Point) и временного свойства `time`.
    * Поддержка форматов: одиночная дата `YYYY-MM-DD` (автоматически расширяется до 24-часового интервала `[date, date + 1 day]`), массив границ `[start_date, end_date]`, массив с шагом дискретизации `[start, end, P1D]`.

2. **Радиометрическая коррекция рельефа (RTC, Radiometric Terrain Correction):**
    * Выполняется на стороне GEE по модели Уландера (Small, 2011) с использованием цифровой модели рельефа Copernicus DEM GLO-30 (`COPERNICUS/DEM/GLO30_2024_1`) с шагом 30 метров.
    * Пересчет удельной эффективной площади рассеяния от площади эллипсоида к истинной площади освещенной поверхности: $\\gamma^0 = \\sigma^0 \\frac{A_\\sigma}{A_\\gamma} = \\sigma^0 \\frac{\\sin \\theta}{\\cos \\psi}$, где $\\theta$ — угол падения луча к эллипсоиду, $\\psi$ — угол между вектором нормали к склону $\\hat{\\mathbf{n}}$ и плоскостью волнового фронта радара.

3. **Временная медианная фильтрация (Temporal Median):**
    * Для каждого снимка формируется выборка всех пролетов за предшествующие 3 месяца.
    * Попиксельное вычисление медианы устраняет случайные атмосферные искажения и сглаживает спекл во времени.

4. **Выгрузка растровых массивов:**
    * Экспорт через прямой шлюз `ee.data.computePixels` в формате `NUMPY_NDARRAY` с пространственным шагом 20 метров.

5. **Пространственная адаптивная фильтрация Gamma MAP:**
    * Байесовский фильтр Maximum A Posteriori в локальном окне $3 \\times 3$ пикселя ($w=3$) с эквивалентным числом некогерентных отсчетов $L=5.0$.
    * Однородные области с коэффициентом вариации $C_I \\le C_u = 1/\\sqrt{5} \\approx 0.4472$ сглаживаются до локального среднего $\\bar{I}$.
    * Точечные цели и резкие границы с $C_I \\ge C_{\\max} = \\sqrt{2/5} \\approx 0.6325$ сохраняются без сглаживания ($\\hat{I} = I$).
    * Промежуточные текстурированные зоны ($C_u < C_I < C_{\\max}$) рассчитываются через аналитическое решение квадратного уравнения с априорным гамма-распределением сигнала.

6. **Логарифмическое масштабирование в децибелы (дБ):**
    * Пересчет каналов мощности: $\\text{VV}_{\\text{dB}} = 10 \\log_{10}(\\sigma^0_{\\text{VV}})$, $\\text{VH}_{\\text{dB}} = 10 \\log_{10}(\\sigma^0_{\\text{VH}})$.
    * Поляризационное отношение: $\\text{ratio}_{\\text{dB}} = \\text{VV}_{\\text{dB}} - \\text{VH}_{\\text{dB}}$.

7. **Формирование NetCDF-4:**
    * Сборка многомерного массива xarray с типами `float32` для радарных переменных и `float64` для географических координат.
"""

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
    max-width: 980px;
    margin: 0 auto;
    padding: 30px;
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

html_body = protect_math(test_md)
full_html = html_template.replace("CONTENT_PLACEHOLDER", html_body)

out_html = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\test_step6.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(full_html)

out_png = r"D:\ПРОФ\ВУЗ\ВТСК\SAR\test_step6.png"
with SB(uc=False, headless=True) as sb:
    sb.open("file:///" + out_html.replace("\\", "/"))
    sb.sleep(2)
    sb.save_screenshot(out_png)

print("Saved test_step6.png")
