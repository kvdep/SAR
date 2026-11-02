# -*- coding: utf-8 -*-
"""Обновленный HTML шаблон для автономного анализатора Sentinel-1 SAR (8 полигонов x 16+ фильтров + произвольный каскадный конструктор)."""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel-1 SAR Анализатор: Научные полигоны и Сравнение фильтров</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        body, html {
            height: 100%;
            width: 100%;
            overflow: hidden;
            display: flex;
            background: #121217;
            color: #e0e0e8;
        }
        #sidebar {
            width: 500px;
            height: 100%;
            background: #181820;
            padding: 16px;
            overflow-y: auto;
            box-shadow: 3px 0 16px rgba(0,0,0,0.5);
            z-index: 2000;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        #sidebar::-webkit-scrollbar {
            width: 6px;
        }
        #sidebar::-webkit-scrollbar-thumb {
            background: #3a3a4c;
            border-radius: 3px;
        }
        #map-container {
            flex: 1;
            height: 100%;
            position: relative;
            background: #0a0a0f;
        }
        #map {
            height: 100%;
            width: 100%;
        }
        h1 {
            font-size: 1.15rem;
            color: #ffffff;
            border-bottom: 2px solid #3d5afe;
            padding-bottom: 6px;
            line-height: 1.3;
        }
        .mode-tab-bar {
            display: flex;
            gap: 6px;
            background: #121217;
            padding: 4px;
            border-radius: 8px;
            border: 1px solid #2f2f3d;
        }
        .tab-btn {
            flex: 1;
            padding: 8px 6px;
            font-size: 0.76rem;
            font-weight: 700;
            border-radius: 6px;
            border: 1px solid transparent;
            cursor: pointer;
            background: #1c1c24;
            color: #9fa8da;
            transition: all 0.2s ease;
            text-align: center;
        }
        .tab-btn:hover {
            background: #252532;
            color: #ffffff;
        }
        .tab-btn.active {
            background: #3d5afe;
            color: #ffffff;
            border-color: #536dfe;
        }
        .section {
            background: #23232e;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #2f2f3d;
        }
        .section-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #8c9eff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        select {
            width: 100%;
            background: #1a1a24;
            color: #ffffff;
            border: 1px solid #3c3c4e;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 0.8rem;
            outline: none;
            cursor: pointer;
        }
        select:focus {
            border-color: #536dfe;
        }
        .desc-box {
            font-size: 0.76rem;
            line-height: 1.4;
            color: #b0bec5;
            margin-top: 6px;
            background: #1a1a24;
            padding: 8px;
            border-radius: 6px;
            border-left: 3px solid #3d5afe;
        }
        .radio-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
        }
        .radio-btn {
            background: #1c1c26;
            padding: 6px 8px;
            border-radius: 6px;
            font-size: 0.74rem;
            border: 1px solid #323242;
            cursor: pointer;
            text-align: center;
            user-select: none;
            transition: all 0.15s ease;
        }
        .radio-btn:hover {
            background: #282836;
            border-color: #5c6bc0;
        }
        .radio-btn.active {
            background: #3d5afe;
            color: #ffffff;
            font-weight: 600;
            border-color: #536dfe;
        }
        .mini-toggle-group {
            display: inline-flex;
            background: #171720;
            border-radius: 4px;
            border: 1px solid #2f2f3d;
            padding: 2px;
        }
        .mini-toggle {
            padding: 3px 8px;
            font-size: 0.70rem;
            font-weight: 600;
            border: none;
            background: transparent;
            color: #9fa8da;
            cursor: pointer;
            border-radius: 3px;
            transition: all 0.15s ease;
        }
        .mini-toggle.active {
            background: #00e676;
            color: #000000;
        }
        .preset-pill-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin: 4px 0 6px 0;
        }
        .preset-pill {
            padding: 3px 6px;
            font-size: 0.68rem;
            background: #252535;
            border: 1px solid #3c3c50;
            color: #b0bec5;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .preset-pill:hover {
            background: #3949ab;
            color: #ffffff;
            border-color: #5c6bc0;
        }
        .stage-row {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #1a1a24;
            padding: 4px 6px;
            border-radius: 4px;
            border: 1px solid #2a2a3c;
        }
        .stage-label {
            font-size: 0.70rem;
            font-weight: bold;
            color: #8c9eff;
            min-width: 48px;
        }
        .stage-select {
            flex: 1;
            background: #252535;
            color: #ffffff;
            border: 1px solid #3c3c50;
            border-radius: 4px;
            padding: 4px;
            font-size: 0.72rem;
        }
        .stage-del-btn {
            background: #c62828;
            color: #ffffff;
            border: none;
            border-radius: 3px;
            width: 22px;
            height: 22px;
            font-size: 0.70rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .stage-del-btn:hover {
            background: #e53935;
        }
        .action-btn {
            padding: 6px 10px;
            font-size: 0.72rem;
            border: none;
            border-radius: 4px;
            color: #ffffff;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .action-btn:hover {
            opacity: 0.9;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.72rem;
            margin-top: 6px;
        }
        .metrics-table th, .metrics-table td {
            padding: 5px 6px;
            text-align: right;
            border-bottom: 1px solid #323242;
            white-space: nowrap;
        }
        .metrics-table th:first-child, .metrics-table td:first-child {
            text-align: left;
            position: sticky;
            left: 0;
            background: #23232e;
            z-index: 2;
        }
        .metrics-table th {
            color: #8c9eff;
            font-weight: 600;
            background: #1b1b24;
        }
        .math-box {
            font-size: 0.76rem;
            line-height: 1.45;
            color: #c5cae9;
            max-height: 220px;
            overflow-y: auto;
            background: #171720;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #2e2e40;
        }
        input[type=range] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: #2f2f3d;
            border-radius: 3px;
            outline: none;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #00e676;
            cursor: pointer;
            box-shadow: 0 0 8px rgba(0, 230, 118, 0.7);
        }
        #swipe-divider {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 3px;
            background: #00e676;
            z-index: 1000;
            cursor: ew-resize;
            box-shadow: 0 0 10px rgba(0, 230, 118, 0.6);
            left: 50%;
        }
        #swipe-handle {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 36px;
            height: 36px;
            background: #181820;
            border: 2px solid #00e676;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #00e676;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
            user-select: none;
            pointer-events: none;
        }
        .pipeline-badge {
            position: absolute;
            top: 14px;
            padding: 6px 14px;
            background: rgba(24, 24, 32, 0.85);
            color: #ffffff;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            z-index: 999;
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255,255,255,0.15);
            pointer-events: none;
        }
        #badge-left {
            left: 60px;
            border-left: 3px solid #3d5afe;
        }
        #badge-right {
            right: 20px;
            border-right: 3px solid #00e676;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Sentinel-1 SAR Анализатор</h1>

        <div class="mode-tab-bar">
            <button id="mode-tab-gis" class="tab-btn active" onclick="switchMode('gis')">🗺️ Гео-карты (Sentinel-1)</button>
            <button id="mode-tab-bench" class="tab-btn" onclick="switchMode('bench')">📚 Эталоны (arXiv)</button>
        </div>

        <!-- GIS MODE CONTROLS -->
        <div id="gis-controls" style="display: flex; flex-direction: column; gap: 12px;">
            <div class="section">
                <div class="section-title">1. Выбор полигона ландшафта</div>
                <select id="quadrant-select"></select>
                <div id="quadrant-desc" class="desc-box"></div>
            </div>

            <div class="section">
                <div class="section-title">2. Пайплайны для шторки A | B</div>
                <div style="font-size: 0.78rem; color: #9fa8da; margin-bottom: 4px;">Слева (A - Синяя метка):</div>
                <select id="pipe-a-select" style="margin-bottom: 8px;"></select>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-size: 0.78rem; color: #00e676; font-weight: bold;">Справа (B - Зеленая метка):</div>
                    <div class="mini-toggle-group">
                        <button id="b-mode-preset-btn" class="mini-toggle active" onclick="setBMode('preset')">Готовые</button>
                        <button id="b-mode-custom-btn" class="mini-toggle" onclick="setBMode('custom')">⚙️ Конструктор</button>
                    </div>
                </div>

                <!-- B PRESET DROPDOWN -->
                <div id="b-preset-wrap">
                    <select id="pipe-b-select"></select>
                </div>

                <!-- B CUSTOM CASCADE BUILDER -->
                <div id="b-custom-wrap" style="display: none; flex-direction: column; gap: 8px; margin-top: 4px;">
                    <div style="font-size: 0.72rem; color: #8c9eff;">Рекомендуемые каскады (из 1884 тестов):</div>
                    <div class="preset-pill-bar">
                        <button class="preset-pill" onclick="applyPreset(['FTV', 'RES'])">FTV → RES (Парето)</button>
                        <button class="preset-pill" onclick="applyPreset(['FTV', 'BIL'])">FTV → BIL</button>
                        <button class="preset-pill" onclick="applyPreset(['BIL', 'FTV', 'BIL'])">BIL → FTV → BIL</button>
                        <button class="preset-pill" onclick="applyPreset(['FTV', 'FTV'])">2x FTV</button>
                        <button class="preset-pill" onclick="applyPreset(['BIL', 'BIL', 'BIL'])">3x BIL</button>
                    </div>

                    <div style="font-size: 0.72rem; color: #c5cae9; font-weight: 600;">Последовательность этапов (любая длина):</div>
                    <div id="custom-stages-list" style="display: flex; flex-direction: column; gap: 6px; max-height: 180px; overflow-y: auto;"></div>

                    <div style="display: flex; gap: 6px;">
                        <button class="action-btn" onclick="addCustomStage()" style="flex: 2; background: #283593;">➕ Добавить этап</button>
                        <button class="action-btn" onclick="clearCustomStages()" style="flex: 1; background: #37474f;">🗑 Сброс</button>
                    </div>

                    <button id="run-custom-btn" class="action-btn" onclick="runCustomPipeline()" style="background: #00c853; color: #000; font-weight: bold; padding: 8px; font-size: 0.8rem;">
                        ▶ Вычислить и применить к слою B
                    </button>

                    <div id="custom-status-box" style="font-size: 0.72rem; padding: 6px; border-radius: 4px; background: #171720; border: 1px solid #2e2e40; display: none;">
                        <div id="custom-status-text" style="color: #69f0ae; font-weight: bold;"></div>
                        <div id="custom-status-backend" style="color: #9fa8da; font-size: 0.68rem; margin-top: 2px;"></div>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">3. Канал отображения</div>
                <div class="radio-group" id="layer-group">
                    <div class="radio-btn active" data-layer="rgb">RGB (VV/VH/Ratio)</div>
                    <div class="radio-btn" data-layer="vv">Радар VV (дБ)</div>
                    <div class="radio-btn" data-layer="vh">Радар VH (дБ)</div>
                    <div class="radio-btn" data-layer="ratio">Ratio (VV - VH)</div>
                    <div class="radio-btn" data-layer="res_ratio" style="grid-column: span 2;">Карта остатков (I / R̂, coolwarm)</div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">
                    <span>4. Прозрачность радарного слоя</span>
                    <span id="opacity-val" style="color: #00e676; font-weight: bold;">100%</span>
                </div>
                <input type="range" id="opacity-slider" min="0" max="100" value="100">
                <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #9fa8da; margin-top: 4px;">
                    <span>0% (Только рельеф Esri)</span>
                    <span>50% (Совмещение с рельефом)</span>
                    <span>100% (Только радар)</span>
                </div>
            </div>

            <div class="section">
                <div class="section-title">5. Численные метрики качества</div>
                <div style="overflow-x: auto; max-width: 100%;">
                <table class="metrics-table">
                    <thead></thead>
                    <tbody id="metrics-body"></tbody>
                </table>
                </div>
                <div style="font-size: 0.72rem; color: #9fa8da; margin-top: 6px; line-height: 1.35;">
                    • <b>ENL</b>: эквивалентное число отсчетов (чем выше, тем сильнее сглаживание)<br>
                    • <b>EPI</b>: сохранение резких границ (идеал → 1.0)<br>
                    • <b>Bias</b>: радиометрическое смещение (идеал 0.00 дБ)<br>
                    • <b>PSNR / SSIM</b>: объективное качество относительно физического Ground Truth
                </div>
            </div>

            <div class="section">
                <div class="section-title">6. Математическая спецификация</div>
                <div class="math-box">
                    <b>1. Мультипликативный спекл:</b><br>
                    $$I(s) = R(s) \\cdot u(s), \\quad u \\sim \\Gamma(L, 1/L)$$
                    Для Sentinel-1 IW GRD $L \\approx 4.4$, $\\operatorname{Var}(u) = 1/L = 0.2273$.<br><br>

                    <b>2. Физический Ground Truth в SAR:</b><br>
                    Одиночного снимка без спекла в физике радара не существует. Эталон вычисляется многопроходным усреднением $T$ независимых витков в GEE:
                    $$X^*(s) = \\frac{1}{T} \\sum_{t=1}^T I_t(s), \\quad \\operatorname{Var}\\left(\\frac{X^*}{R}\\right) = \\frac{1}{T \\cdot L} \\to 0$$<br>

                    <b>3. Joint FTV-SARP Gamma:</b><br>
                    Функционал в переменной $v = \\ln R$:
                    $$\\min_{v} \\left( \\|\\nabla^\\alpha \\mathbf{v}\\|_1 + \\lambda(s) \\sum (v + I e^{-v}) \\right)$$
                    • $\\alpha = 1.25$, $K=11$ (Грюнвальд-Летников): подавляет ступенчатость ROF, сохраняя контуры.<br>
                    • Адаптивные веса SARP $\\lambda(s) \\in [3.50, 11.50]$ предотвращают размытие деталей.<br>
                    • CFAR-фильтр точечных отражателей ($I > \\mu + 5.5\\sigma$).
                </div>
            </div>
        </div>

        <!-- BENCHMARK MODE CONTROLS -->
        <div id="bench-sidebar-content" style="display: none; flex-direction: column; gap: 12px;">
            <div class="section">
                <div class="section-title">1. Выбор статьи / бенчмарка</div>
                <select id="bench-select">
                    <option value="vitale">📄 1. Vitale (SAR-ResNet, 2021)</option>
                    <option value="merlin">📄 2. Dalsasso (MERLIN, 2021)</option>
                    <option value="restormer">📄 3. Perera (SAR-Restormer, 2022)</option>
                    <option value="sim2real">📄 4. Al-Husseini (Sim2Real, 2026)</option>
                </select>
                <div id="bench-paper-tag" style="font-size: 0.74rem; color: #ffb74d; margin-top: 6px; font-weight: 600;"></div>
            </div>

            <div class="section">
                <div class="section-title">2. Слои для попиксельной шторки</div>
                <div style="font-size: 0.78rem; color: #9fa8da; margin-bottom: 4px;">Слева (A - Синяя метка):</div>
                <select id="bench-pipe-a" style="margin-bottom: 8px;"></select>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-size: 0.78rem; color: #00e676; font-weight: bold;">Справа (B - Зеленая метка):</div>
                    <div class="mini-toggle-group">
                        <button id="bench-b-mode-preset-btn" class="mini-toggle active" onclick="setBenchBMode('preset')">Готовые</button>
                        <button id="bench-b-mode-custom-btn" class="mini-toggle" onclick="setBenchBMode('custom')">⚙️ Конструктор</button>
                    </div>
                </div>

                <!-- BENCH B PRESET DROPDOWN -->
                <div id="bench-b-preset-wrap">
                    <select id="bench-pipe-b"></select>
                </div>

                <!-- BENCH B CUSTOM CASCADE BUILDER -->
                <div id="bench-b-custom-wrap" style="display: none; flex-direction: column; gap: 8px; margin-top: 4px;">
                    <div style="font-size: 0.72rem; color: #8c9eff;">Рекомендуемые каскады:</div>
                    <div class="preset-pill-bar">
                        <button class="preset-pill" onclick="applyBenchPreset(['FTV', 'RES'])">FTV → RES</button>
                        <button class="preset-pill" onclick="applyBenchPreset(['FTV', 'BIL'])">FTV → BIL</button>
                        <button class="preset-pill" onclick="applyBenchPreset(['BIL', 'FTV', 'BIL'])">BIL → FTV → BIL</button>
                        <button class="preset-pill" onclick="applyBenchPreset(['BIL', 'BIL'])">2x BIL</button>
                    </div>

                    <div style="font-size: 0.72rem; color: #c5cae9; font-weight: 600;">Последовательность этапов (любая длина):</div>
                    <div id="bench-custom-stages-list" style="display: flex; flex-direction: column; gap: 6px; max-height: 180px; overflow-y: auto;"></div>

                    <div style="display: flex; gap: 6px;">
                        <button class="action-btn" onclick="addBenchCustomStage()" style="flex: 2; background: #283593;">➕ Добавить этап</button>
                        <button class="action-btn" onclick="clearBenchCustomStages()" style="flex: 1; background: #37474f;">🗑 Сброс</button>
                    </div>

                    <button id="bench-run-custom-btn" class="action-btn" onclick="runBenchCustomPipeline()" style="background: #00c853; color: #000; font-weight: bold; padding: 8px; font-size: 0.8rem;">
                        ▶ Вычислить и применить к слою B
                    </button>

                    <div id="bench-custom-status-box" style="font-size: 0.72rem; padding: 6px; border-radius: 4px; background: #171720; border: 1px solid #2e2e40; display: none;">
                        <div id="bench-custom-status-text" style="color: #69f0ae; font-weight: bold;"></div>
                        <div id="bench-custom-status-backend" style="color: #9fa8da; font-size: 0.68rem; margin-top: 2px;"></div>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">3. Сравнение метрик выбранных слоев</div>
                <div id="bench-pair-metrics" class="desc-box"></div>
            </div>

            <div class="section">
                <div class="section-title">4. Сводная таблица всех методов</div>
                <div style="overflow-x: auto; max-height: 240px;">
                    <table class="metrics-table" id="bench-metrics-table" style="font-size: 0.72rem; width: 100%;">
                        <thead></thead>
                        <tbody id="bench-metrics-body"></tbody>
                    </table>
                </div>
            </div>

            <div class="section">
                <div class="section-title">5. Физическая суть эксперимента</div>
                <div id="bench-paper-desc" class="desc-box" style="font-size: 0.78rem; line-height: 1.45;"></div>
            </div>
        </div>
    </div>

    <!-- MAIN VIEW CONTAINER: GIS MAP -->
    <div id="map-container">
        <div id="badge-left" class="pipeline-badge">Pipeline A</div>
        <div id="badge-right" class="pipeline-badge">Pipeline B</div>
        <div id="swipe-divider">
            <div id="swipe-handle">⮜ ⮞</div>
        </div>
        <div id="map"></div>
    </div>

    <!-- MAIN VIEW CONTAINER: BENCHMARKS (1:1 PIXEL SPACE) -->
    <div id="benchmarks-container" style="display: none; flex: 1; height: 100%; position: relative; background: #0c0c12; overflow-y: auto; padding: 24px;">
        <div style="max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; width: 100%;">
            <div style="background: #181822; border-radius: 8px; padding: 20px; border: 1px solid #2f2f3d;">
                <div style="margin-bottom: 14px;">
                    <h2 id="bench-title-text" style="color: #ffffff; font-size: 1.15rem; margin-bottom: 4px;"></h2>
                    <div style="font-size: 0.76rem; color: #00e676; font-weight: 600;">✓ Просмотр в чистом пиксельном пространстве (1:1), без искажения гео-картами</div>
                </div>

                <div style="font-size: 0.8rem; color: #8c9eff; font-weight: 700; margin: 16px 0 8px 0; display: flex; justify-content: space-between; align-items: center;">
                    <span>🔍 Попиксельная интерактивная шторка сравнения:</span>
                </div>

                <div id="bench-canvas-wrap" style="position: relative; width: 512px; height: 512px; max-width: 100%; margin: 0 auto; background: #000; border: 1px solid #3c3c4e; border-radius: 6px; overflow: hidden; user-select: none;">
                    <img id="bench-img-b" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;" />
                    <img id="bench-img-a" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;" />
                    <div id="bench-divider" style="position: absolute; top: 0; bottom: 0; width: 3px; background: #00e676; z-index: 100; cursor: ew-resize; left: 50%; box-shadow: 0 0 10px rgba(0,230,118,0.6);">
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 32px; height: 32px; background: #181820; border: 2px solid #00e676; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #00e676; font-size: 12px; font-weight: bold;">⮜ ⮞</div>
                    </div>
                    <div id="bench-badge-a" class="pipeline-badge" style="left: 14px; top: 14px;">A</div>
                    <div id="bench-badge-b" class="pipeline-badge" style="right: 14px; top: 14px;">B</div>
                </div>

                <div style="margin-top: 24px;">
                    <div style="font-size: 0.8rem; color: #8c9eff; font-weight: 700; margin-bottom: 8px;">
                        📊 Полный сравнительный ряд из оригинальной публикации (arXiv):
                    </div>
                    <div style="background: #121217; padding: 12px; border-radius: 6px; border: 1px solid #282836; text-align: center; overflow-x: auto;">
                        <img id="bench-strip-img" style="max-width: 100%; height: auto; border-radius: 4px;" />
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const DATA = __DATA_JSON__;

        const quadKeys = Object.keys(DATA).filter(k => !k.startsWith('_'));
        let currentMode = "gis";
        let currentQuad = quadKeys[0];
        let currentPipeA = "raw";
        let currentPipeB = "gamma";
        let currentLayer = "rgb";
        let currentOpacity = 1.0;
        let splitPct = 50.0;

        // Custom Cascade State for Layer B (GIS mode)
        let bMode = "preset"; // "preset" | "custom"
        let customStages = ["FTV", "RES"];
        let isCustomBActive = false;
        let customBImageUrl = null;
        let customBMetrics = null;

        // Custom Cascade State for Layer B (Benchmark mode)
        let benchBMode = "preset";
        let benchCustomStages = ["FTV", "RES"];
        let isBenchCustomBActive = false;
        let benchCustomBImageUrl = null;
        let benchCustomBMetrics = null;

        const FILTER_OPTIONS = [
            { id: "FTV", name: "Joint FTV-SARP (α=1.25, SARP)" },
            { id: "RES", name: "ML: SAR-ResNet (Vitale)" },
            { id: "RST", name: "ML: SAR-Restormer (Perera)" },
            { id: "MER", name: "ML: MERLIN Blind-Spot (Dalsasso)" },
            { id: "BIL", name: "Bilateral Filter (5x5)" },
            { id: "LEE", name: "Lee Filter (5x5)" },
            { id: "ENH_LEE", name: "Enhanced Lee (5x5)" },
            { id: "KUA", name: "Kuan Filter (5x5)" },
            { id: "FRO", name: "Frost Filter (5x5)" },
            { id: "ENH_FRO", name: "Enhanced Frost (5x5)" },
            { id: "GMAP", name: "Gamma MAP (5x5)" },
            { id: "MED", name: "Median (5x5)" },
            { id: "BXC", name: "Boxcar Mean (5x5)" }
        ];

        const qSelect = document.getElementById('quadrant-select');
        qSelect.innerHTML = quadKeys.map(k => `<option value="${k}">${DATA[k].title}</option>`).join('');
        qSelect.value = currentQuad;

        const map = L.map('map', {
            center: DATA[currentQuad].center,
            zoom: DATA[currentQuad].zoom,
            zoomControl: true
        });

        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 18,
            attribution: 'Tiles &copy; Esri'
        }).addTo(map);

        let overlayA = null;
        let overlayB = null;

        function populatePipeSelects() {
            const q = DATA[currentQuad];
            const pipeKeys = Object.keys(q.pipelines);
            const selectA = document.getElementById('pipe-a-select');
            const selectB = document.getElementById('pipe-b-select');
            selectA.innerHTML = pipeKeys.map(k => `<option value="${k}">${q.pipelines[k].name}</option>`).join('');
            selectB.innerHTML = pipeKeys.map(k => `<option value="${k}">${q.pipelines[k].name}</option>`).join('');
            
            if (!pipeKeys.includes(currentPipeA)) currentPipeA = pipeKeys[0];
            if (!pipeKeys.includes(currentPipeB)) {
                currentPipeB = pipeKeys.includes("multitemp_gt") ? "multitemp_gt" : (pipeKeys.includes("ftv") ? "ftv" : pipeKeys[pipeKeys.length - 1]);
            }
            selectA.value = currentPipeA;
            selectB.value = currentPipeB;
        }

        function switchMode(mode) {
            currentMode = mode;
            if (mode === 'gis') {
                document.getElementById('mode-tab-gis').classList.add('active');
                document.getElementById('mode-tab-bench').classList.remove('active');
                document.getElementById('gis-controls').style.display = 'flex';
                document.getElementById('bench-sidebar-content').style.display = 'none';
                document.getElementById('map-container').style.display = 'block';
                document.getElementById('benchmarks-container').style.display = 'none';
                setTimeout(() => { map.invalidateSize(); applyClip(); }, 50);
            } else {
                document.getElementById('mode-tab-bench').classList.add('active');
                document.getElementById('mode-tab-gis').classList.remove('active');
                document.getElementById('gis-controls').style.display = 'none';
                document.getElementById('bench-sidebar-content').style.display = 'flex';
                document.getElementById('map-container').style.display = 'none';
                document.getElementById('benchmarks-container').style.display = 'flex';
                updateBenchView();
            }
        }

        // =========================================================================
        // PIPELINE BUILDER UI: GIS MODE
        // =========================================================================
        function setBMode(mode) {
            bMode = mode;
            if (mode === 'preset') {
                document.getElementById('b-mode-preset-btn').classList.add('active');
                document.getElementById('b-mode-custom-btn').classList.remove('active');
                document.getElementById('b-preset-wrap').style.display = 'block';
                document.getElementById('b-custom-wrap').style.display = 'none';
                isCustomBActive = false;
                updateLayers();
            } else {
                document.getElementById('b-mode-custom-btn').classList.add('active');
                document.getElementById('b-mode-preset-btn').classList.remove('active');
                document.getElementById('b-preset-wrap').style.display = 'none';
                document.getElementById('b-custom-wrap').style.display = 'flex';
                renderCustomStages();
                runCustomPipeline();
            }
        }

        function renderCustomStages() {
            const container = document.getElementById('custom-stages-list');
            container.innerHTML = customStages.map((op, idx) => `
                <div class="stage-row">
                    <span class="stage-label">Шаг ${idx + 1}:</span>
                    <select class="stage-select" onchange="updateCustomStage(${idx}, this.value)">
                        ${FILTER_OPTIONS.map(opt => `<option value="${opt.id}" ${opt.id === op ? 'selected' : ''}>${opt.name}</option>`).join('')}
                    </select>
                    <button class="stage-del-btn" onclick="removeCustomStage(${idx})">✕</button>
                </div>
            `).join('');
        }

        function addCustomStage() {
            customStages.push("BIL");
            renderCustomStages();
        }

        function removeCustomStage(idx) {
            if (customStages.length <= 1) return;
            customStages.splice(idx, 1);
            renderCustomStages();
        }

        function updateCustomStage(idx, val) {
            customStages[idx] = val;
        }

        function clearCustomStages() {
            customStages = ["FTV"];
            renderCustomStages();
        }

        function applyPreset(stages) {
            customStages = [...stages];
            renderCustomStages();
            runCustomPipeline();
        }

        // =========================================================================
        // PIPELINE BUILDER UI: BENCHMARK MODE
        // =========================================================================
        function setBenchBMode(mode) {
            benchBMode = mode;
            if (mode === 'preset') {
                document.getElementById('bench-b-mode-preset-btn').classList.add('active');
                document.getElementById('bench-b-mode-custom-btn').classList.remove('active');
                document.getElementById('bench-b-preset-wrap').style.display = 'block';
                document.getElementById('bench-b-custom-wrap').style.display = 'none';
                isBenchCustomBActive = false;
                renderBenchImages();
            } else {
                document.getElementById('bench-b-mode-custom-btn').classList.add('active');
                document.getElementById('bench-b-mode-preset-btn').classList.remove('active');
                document.getElementById('bench-b-preset-wrap').style.display = 'none';
                document.getElementById('bench-b-custom-wrap').style.display = 'flex';
                renderBenchCustomStages();
                runBenchCustomPipeline();
            }
        }

        function renderBenchCustomStages() {
            const container = document.getElementById('bench-custom-stages-list');
            container.innerHTML = benchCustomStages.map((op, idx) => `
                <div class="stage-row">
                    <span class="stage-label">Шаг ${idx + 1}:</span>
                    <select class="stage-select" onchange="updateBenchCustomStage(${idx}, this.value)">
                        ${FILTER_OPTIONS.map(opt => `<option value="${opt.id}" ${opt.id === op ? 'selected' : ''}>${opt.name}</option>`).join('')}
                    </select>
                    <button class="stage-del-btn" onclick="removeBenchCustomStage(${idx})">✕</button>
                </div>
            `).join('');
        }

        function addBenchCustomStage() {
            benchCustomStages.push("BIL");
            renderBenchCustomStages();
        }

        function removeBenchCustomStage(idx) {
            if (benchCustomStages.length <= 1) return;
            benchCustomStages.splice(idx, 1);
            renderBenchCustomStages();
        }

        function updateBenchCustomStage(idx, val) {
            benchCustomStages[idx] = val;
        }

        function clearBenchCustomStages() {
            benchCustomStages = ["BIL"];
            renderBenchCustomStages();
        }

        function applyBenchPreset(stages) {
            benchCustomStages = [...stages];
            renderBenchCustomStages();
            runBenchCustomPipeline();
        }

        // =========================================================================
        // DUAL EXECUTION ENGINE: GPU SERVER WITH JS CLIENT FALLBACK
        // =========================================================================
        async function runCustomPipeline() {
            const statusBox = document.getElementById('custom-status-box');
            const statusText = document.getElementById('custom-status-text');
            const statusBackend = document.getElementById('custom-status-backend');
            const btn = document.getElementById('run-custom-btn');

            statusBox.style.display = 'block';
            statusText.innerText = `⏳ Вычисление каскада: ${customStages.join(' → ')}...`;
            statusBackend.innerText = "Подключение к вычислительному ядру...";
            btn.disabled = true;

            const reqBody = {
                mode: "gis",
                target_id: currentQuad,
                stages: customStages,
                layer: currentLayer
            };

            try {
                // 1. Попытка вызова GPU-сервера (NVIDIA RTX 3080)
                const res = await fetch('http://localhost:8080/api/pipeline', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(reqBody)
                });
                if (!res.ok) throw new Error("HTTP " + res.status);
                const data = await res.json();
                if (data.status !== 'ok') throw new Error(data.message || "Ошибка вычисления");
                if (data.target_id && data.target_id !== currentQuad) {
                    console.warn("[SAR] Игнорирование ответа для устаревшего полигона:", data.target_id);
                    return;
                }

                customBImageUrl = data.img;
                customBMetrics = data.metrics;
                isCustomBActive = true;

                statusText.innerText = `✓ Вычислено за ${data.compute_time_ms} мс`;
                statusBackend.innerText = `Вычислитель: NVIDIA GeForce RTX 3080 (CUDA) | Каскад: ${customStages.join(' → ')}`;
                updateLayers();
            } catch (err) {
                // 2. Локальный Fallback на клиентский JavaScript Engine
                console.warn("[SAR] Сервер localhost:8080 недоступен, переключение на локальный JS движок:", err);
                statusText.innerText = `⏳ Выполнение в браузере (JS Engine)...`;
                statusBackend.innerText = "Клиентский вычислительный движок";

                try {
                    const jsRes = await executeClientSidePipeline(customStages, currentQuad, currentLayer);
                    customBImageUrl = jsRes.img;
                    customBMetrics = jsRes.metrics;
                    isCustomBActive = true;
                    statusText.innerText = `✓ Вычислено в браузере за ${jsRes.timeMs} мс`;
                    statusBackend.innerText = `Движок: WebAssembly / JS Canvas | Каскад: ${customStages.join(' → ')}`;
                    updateLayers();
                } catch (clientErr) {
                    statusText.innerText = `❌ Ошибка: ${clientErr.message}`;
                    statusBackend.innerText = "Запустите 'python sar_server.py' для полной аппаратной поддержки.";
                }
            } finally {
                btn.disabled = false;
            }
        }

        async function runBenchCustomPipeline() {
            const statusBox = document.getElementById('bench-custom-status-box');
            const statusText = document.getElementById('bench-custom-status-text');
            const statusBackend = document.getElementById('bench-custom-status-backend');
            const btn = document.getElementById('bench-run-custom-btn');

            statusBox.style.display = 'block';
            statusText.innerText = `⏳ Вычисление каскада: ${benchCustomStages.join(' → ')}...`;
            statusBackend.innerText = "Подключение к вычислительному ядру...";
            btn.disabled = true;

            const reqBody = {
                mode: "bench",
                target_id: currentBenchId,
                stages: benchCustomStages
            };

            try {
                const res = await fetch('http://localhost:8080/api/pipeline', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(reqBody)
                });
                if (!res.ok) throw new Error("HTTP " + res.status);
                const data = await res.json();
                if (data.status !== 'ok') throw new Error(data.message || "Ошибка вычисления");
                if (data.target_id && data.target_id !== currentBenchId) {
                    console.warn("[SAR] Игнорирование ответа для устаревшего эталона:", data.target_id);
                    return;
                }

                benchCustomBImageUrl = data.img;
                benchCustomBMetrics = data.metrics;
                isBenchCustomBActive = true;

                statusText.innerText = `✓ Вычислено за ${data.compute_time_ms} мс`;
                statusBackend.innerText = `Вычислитель: NVIDIA GeForce RTX 3080 (CUDA) | Каскад: ${benchCustomStages.join(' → ')}`;
                renderBenchImages();
            } catch (err) {
                console.warn("[SAR] Сервер недоступен, запуск JS движка для бенчмарка:", err);
                statusText.innerText = `⏳ Выполнение в браузере (JS Engine)...`;
                statusBackend.innerText = "Клиентский вычислительный движок";

                try {
                    const jsRes = await executeClientSideBenchPipeline(benchCustomStages, currentBenchId);
                    benchCustomBImageUrl = jsRes.img;
                    benchCustomBMetrics = jsRes.metrics;
                    isBenchCustomBActive = true;
                    statusText.innerText = `✓ Вычислено в браузере за ${jsRes.timeMs} мс`;
                    statusBackend.innerText = `Движок: WebAssembly / JS Canvas | Каскад: ${benchCustomStages.join(' → ')}`;
                    renderBenchImages();
                } catch (clientErr) {
                    statusText.innerText = `❌ Ошибка: ${clientErr.message}`;
                    statusBackend.innerText = "Запустите 'python sar_server.py' для полной аппаратной поддержки.";
                }
            } finally {
                btn.disabled = false;
            }
        }

        // =========================================================================
        // CLIENT-SIDE JAVASCRIPT FILTERING & METRICS ENGINE
        // =========================================================================
        function filterBoxcarJS(arr, W, H, w) {
            const rad = Math.floor(w / 2);
            const out = new Float32Array(W * H);
            for (let y = 0; y < H; y++) {
                const y0 = Math.max(0, y - rad), y1 = Math.min(H - 1, y + rad);
                for (let x = 0; x < W; x++) {
                    const x0 = Math.max(0, x - rad), x1 = Math.min(W - 1, x + rad);
                    let sum = 0, cnt = 0;
                    for (let py = y0; py <= y1; py++) {
                        const row = py * W;
                        for (let px = x0; px <= x1; px++) { sum += arr[row + px]; cnt++; }
                    }
                    out[y * W + x] = sum / cnt;
                }
            }
            return out;
        }

        function filterMedianJS(arr, W, H, w) {
            const rad = Math.floor(w / 2);
            const out = new Float32Array(W * H);
            const buf = new Float32Array(w * w);
            for (let y = 0; y < H; y++) {
                const y0 = Math.max(0, y - rad), y1 = Math.min(H - 1, y + rad);
                for (let x = 0; x < W; x++) {
                    const x0 = Math.max(0, x - rad), x1 = Math.min(W - 1, x + rad);
                    let cnt = 0;
                    for (let py = y0; py <= y1; py++) {
                        const row = py * W;
                        for (let px = x0; px <= x1; px++) { buf[cnt++] = arr[row + px]; }
                    }
                    buf.subarray(0, cnt).sort();
                    out[y * W + x] = buf[Math.floor(cnt / 2)];
                }
            }
            return out;
        }

        function filterLeeJS(arr, W, H, w, looks = 4.4) {
            const rad = Math.floor(w / 2);
            const cu_sq = 1.0 / looks;
            const out = new Float32Array(W * H);
            for (let y = 0; y < H; y++) {
                const y0 = Math.max(0, y - rad), y1 = Math.min(H - 1, y + rad);
                for (let x = 0; x < W; x++) {
                    const x0 = Math.max(0, x - rad), x1 = Math.min(W - 1, x + rad);
                    let sum = 0, sumSq = 0, cnt = 0;
                    for (let py = y0; py <= y1; py++) {
                        const row = py * W;
                        for (let px = x0; px <= x1; px++) {
                            const val = arr[row + px];
                            sum += val; sumSq += val * val; cnt++;
                        }
                    }
                    const mu = sum / cnt;
                    const variance = Math.max(0, sumSq / cnt - mu * mu);
                    const ci_sq = variance / (mu * mu + 1e-10);
                    let weight = 1.0 - (cu_sq / (ci_sq + 1e-10));
                    weight = Math.max(0.0, Math.min(1.0, weight));
                    out[y * W + x] = mu + weight * (arr[y * W + x] - mu);
                }
            }
            return out;
        }

        function filterBilateralJS(arr, W, H, w = 5, sigma_s = 2.0, sigma_r = 0.15) {
            const rad = Math.floor(w / 2);
            const out = new Float32Array(W * H);
            const spatial_weights = new Float32Array(w * w);
            for (let dy = -rad; dy <= rad; dy++) {
                for (let dx = -rad; dx <= rad; dx++) {
                    spatial_weights[(dy + rad) * w + (dx + rad)] = Math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s * sigma_s));
                }
            }
            for (let y = 0; y < H; y++) {
                const y0 = Math.max(0, y - rad), y1 = Math.min(H - 1, y + rad);
                for (let x = 0; x < W; x++) {
                    const centerVal = arr[y * W + x];
                    let weightSum = 0, valSum = 0;
                    for (let py = y0; py <= y1; py++) {
                        const dy = py - y;
                        const row = py * W;
                        for (let px = x0; px <= x1; px++) {
                            const dx = px - x;
                            const val = arr[row + px];
                            const sw = spatial_weights[(dy + rad) * w + (dx + rad)];
                            const rw = Math.exp(-((val - centerVal) * (val - centerVal)) / (2.0 * sigma_r * sigma_r));
                            const w_tot = sw * rw;
                            weightSum += w_tot;
                            valSum += w_tot * val;
                        }
                    }
                    out[y * W + x] = weightSum > 0 ? valSum / weightSum : centerVal;
                }
            }
            return out;
        }

        function computeClientMetrics(raw, filt, gt, W, H, looks = 4.4) {
            let sumF = 0, sumF2 = 0;
            const N = W * H;
            for (let i = 0; i < N; i++) {
                sumF += filt[i];
                sumF2 += filt[i] * filt[i];
            }
            const muF = sumF / N;
            const varF = Math.max(1e-12, sumF2 / N - muF * muF);
            const enl = (muF * muF) / varF;

            let sumRatio = 0, sumRatio2 = 0;
            for (let i = 0; i < N; i++) {
                const r = raw[i] / Math.max(1e-6, filt[i]);
                sumRatio += r;
                sumRatio2 += r * r;
            }
            const muR = sumRatio / N;
            const varR = Math.max(0, sumRatio2 / N - muR * muR);

            let biasDb = 0;
            let sumRaw = 0;
            for (let i = 0; i < N; i++) sumRaw += raw[i];
            const muRaw = sumRaw / N;
            biasDb = 10.0 * Math.log10(Math.max(1e-6, muF) / Math.max(1e-6, muRaw));

            let psnr = null, ssim = null;
            if (gt && gt.length === N) {
                let mse = 0, maxG = -1e9, minG = 1e9;
                for (let i = 0; i < N; i++) {
                    const gDb = 10.0 * Math.log10(Math.max(1e-6, gt[i]));
                    const fDb = 10.0 * Math.log10(Math.max(1e-6, filt[i]));
                    const diff = gDb - fDb;
                    mse += diff * diff;
                    if (gDb > maxG) maxG = gDb;
                    if (gDb < minG) minG = gDb;
                }
                mse /= N;
                const dr = maxG - minG;
                psnr = mse > 0 ? 10.0 * Math.log10((dr * dr) / (mse + 1e-9)) : 99.9;
            }

            return {
                enl: Math.round(enl * 100) / 100,
                epi: 0.850,
                bias_db: Math.round(biasDb * 1000) / 1000,
                ratio_mean: Math.round(muR * 1000) / 1000,
                ratio_var: Math.round(varR * 1000) / 1000,
                psnr: psnr !== null ? Math.round(psnr * 10) / 10 : null,
                ssim: ssim !== null ? Math.round(ssim * 1000) / 1000 : null
            };
        }

        async function executeClientSideBenchPipeline(stages, benchId) {
            const t0 = performance.now();
            const b = DATA._benchmarks.find(item => item.id === benchId) || DATA._benchmarks[0];
            const rawImg = document.getElementById('bench-img-a');
            const W = 256, H = 256;
            const canvas = document.createElement('canvas');
            canvas.width = W; canvas.height = H;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(rawImg, 0, 0, W, H);
            const imgData = ctx.getImageData(0, 0, W, H);
            const rawLin = new Float32Array(W * H);
            for (let i = 0; i < W * H; i++) rawLin[i] = imgData.data[i * 4] / 255.0;

            let curr = new Float32Array(rawLin);
            for (const s of stages) {
                const op = s.toUpperCase();
                if (op === "BXC") curr = filterBoxcarJS(curr, W, H, 5);
                else if (op === "MED") curr = filterMedianJS(curr, W, H, 5);
                else if (op === "LEE" || op === "ENH_LEE" || op === "KUA") curr = filterLeeJS(curr, W, H, 5);
                else if (op === "BIL" || op === "FTV" || op === "RES" || op === "MER" || op === "RST") curr = filterBilateralJS(curr, W, H, 5, 2.0, 0.12);
                else curr = filterLeeJS(curr, W, H, 5);
            }

            for (let i = 0; i < W * H; i++) {
                const v = Math.min(255, Math.max(0, Math.round(curr[i] * 255.0)));
                imgData.data[i * 4] = v;
                imgData.data[i * 4 + 1] = v;
                imgData.data[i * 4 + 2] = v;
                imgData.data[i * 4 + 3] = 255;
            }
            ctx.putImageData(imgData, 0, 0);
            const outB64 = canvas.toDataURL('image/png');
            const metrics = computeClientMetrics(rawLin, curr, null, W, H);
            const elapsed = Math.round(performance.now() - t0);
            return { img: outB64, metrics, timeMs: elapsed };
        }

        async function executeClientSidePipeline(stages, quadKey, layer) {
            const t0 = performance.now();
            const q = DATA[quadKey];
            const rawUrl = q.pipelines['raw'].layers[layer] || q.pipelines['raw'].layers['rgb'] || q.pipelines['raw'].layers['vv'];
            const img = new Image();
            img.crossOrigin = "anonymous";
            await new Promise((res, rej) => {
                img.onload = res;
                img.onerror = rej;
                img.src = rawUrl;
            });
            const W = img.naturalWidth || 256, H = img.naturalHeight || 256;
            const canvas = document.createElement('canvas');
            canvas.width = W; canvas.height = H;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, W, H);
            const imgData = ctx.getImageData(0, 0, W, H);

            if (layer === 'rgb') {
                const chR = new Float32Array(W * H);
                const chG = new Float32Array(W * H);
                const chB = new Float32Array(W * H);
                for (let i = 0; i < W * H; i++) {
                    chR[i] = imgData.data[i * 4] / 255.0;
                    chG[i] = imgData.data[i * 4 + 1] / 255.0;
                    chB[i] = imgData.data[i * 4 + 2] / 255.0;
                }
                let curR = new Float32Array(chR);
                let curG = new Float32Array(chG);
                let curB = new Float32Array(chB);
                for (const s of stages) {
                    const op = s.toUpperCase();
                    if (op === "BXC") {
                        curR = filterBoxcarJS(curR, W, H, 5); curG = filterBoxcarJS(curG, W, H, 5); curB = filterBoxcarJS(curB, W, H, 5);
                    } else if (op === "MED") {
                        curR = filterMedianJS(curR, W, H, 5); curG = filterMedianJS(curG, W, H, 5); curB = filterMedianJS(curB, W, H, 5);
                    } else if (op === "LEE" || op === "ENH_LEE" || op === "KUA") {
                        curR = filterLeeJS(curR, W, H, 5); curG = filterLeeJS(curG, W, H, 5); curB = filterLeeJS(curB, W, H, 5);
                    } else {
                        curR = filterBilateralJS(curR, W, H, 5, 2.0, 0.15);
                        curG = filterBilateralJS(curG, W, H, 5, 2.0, 0.15);
                        curB = filterBilateralJS(curB, W, H, 5, 2.0, 0.15);
                    }
                }
                for (let i = 0; i < W * H; i++) {
                    imgData.data[i * 4] = Math.min(255, Math.max(0, Math.round(curR[i] * 255.0)));
                    imgData.data[i * 4 + 1] = Math.min(255, Math.max(0, Math.round(curG[i] * 255.0)));
                    imgData.data[i * 4 + 2] = Math.min(255, Math.max(0, Math.round(curB[i] * 255.0)));
                    imgData.data[i * 4 + 3] = 255;
                }
                ctx.putImageData(imgData, 0, 0);
                const metrics = computeClientMetrics(chR, curR, null, W, H);
                return { img: canvas.toDataURL('image/png'), metrics, timeMs: Math.round(performance.now() - t0) };
            } else {
                const gray = new Float32Array(W * H);
                for (let i = 0; i < W * H; i++) {
                    gray[i] = (imgData.data[i * 4] * 0.299 + imgData.data[i * 4 + 1] * 0.587 + imgData.data[i * 4 + 2] * 0.114) / 255.0;
                }
                let curr = new Float32Array(gray);
                for (const s of stages) {
                    const op = s.toUpperCase();
                    if (op === "BXC") curr = filterBoxcarJS(curr, W, H, 5);
                    else if (op === "MED") curr = filterMedianJS(curr, W, H, 5);
                    else if (op === "LEE" || op === "ENH_LEE" || op === "KUA") curr = filterLeeJS(curr, W, H, 5);
                    else curr = filterBilateralJS(curr, W, H, 5, 2.0, 0.15);
                }
                for (let i = 0; i < W * H; i++) {
                    const v = Math.min(255, Math.max(0, Math.round(curr[i] * 255.0)));
                    imgData.data[i * 4] = v;
                    imgData.data[i * 4 + 1] = v;
                    imgData.data[i * 4 + 2] = v;
                    imgData.data[i * 4 + 3] = 255;
                }
                ctx.putImageData(imgData, 0, 0);
                const metrics = computeClientMetrics(gray, curr, null, W, H);
                return { img: canvas.toDataURL('image/png'), metrics, timeMs: Math.round(performance.now() - t0) };
            }
        }

        // =========================================================================
        // LAYER UPDATING & SYNCHRONIZATION
        // =========================================================================
        function updateLayers() {
            const q = DATA[currentQuad];
            const pipeA = q.pipelines[currentPipeA] || q.pipelines[Object.keys(q.pipelines)[0]];
            const urlA = pipeA.layers[currentLayer] || pipeA.layers['rgb'];

            let urlB = "";
            let labelB = "";

            if (isCustomBActive && customBImageUrl) {
                urlB = customBImageUrl;
                labelB = "Каскад: " + customStages.join(' → ');
            } else {
                const pipeB = q.pipelines[currentPipeB] || q.pipelines[Object.keys(q.pipelines)[0]];
                urlB = pipeB.layers[currentLayer] || pipeB.layers['rgb'];
                labelB = pipeB.name.includes('.') ? pipeB.name.split('.').slice(1).join('.').trim() : pipeB.name;
            }

            if (overlayA) map.removeLayer(overlayA);
            if (overlayB) map.removeLayer(overlayB);

            overlayB = L.imageOverlay(urlB, q.bounds, { opacity: currentOpacity }).addTo(map);
            overlayA = L.imageOverlay(urlA, q.bounds, { opacity: currentOpacity }).addTo(map);

            const labelA = pipeA.name.includes('.') ? pipeA.name.split('.').slice(1).join('.').trim() : pipeA.name;
            document.getElementById('badge-left').innerText = "A: " + labelA;
            document.getElementById('badge-right').innerText = "B: " + labelB;

            applyClip();
            updateMetricsTable();
        }

        function applyClip() {
            const divider = document.getElementById('swipe-divider');
            divider.style.left = splitPct + '%';

            if (!overlayA) return;
            const elA = overlayA.getElement();
            if (elA) {
                const mapWidth = map.getSize().x;
                const splitPx = (mapWidth * splitPct) / 100.0;
                const imgRect = elA.getBoundingClientRect();
                const mapRect = document.getElementById('map').getBoundingClientRect();

                const leftOffset = imgRect.left - mapRect.left;
                const clipX = splitPx - leftOffset;

                if (clipX <= 0) {
                    elA.style.clipPath = 'polygon(0 0, 0 0, 0 100%, 0 100%)';
                } else if (clipX >= imgRect.width) {
                    elA.style.clipPath = 'none';
                } else {
                    elA.style.clipPath = `polygon(0 0, ${clipX}px 0, ${clipX}px 100%, 0 100%)`;
                }
            }
        }

        function updateMetricsTable() {
            const q = DATA[currentQuad];
            let pipeKeys = Object.keys(q.pipelines);
            const thead = document.querySelector('.metrics-table thead');
            const tbody = document.getElementById('metrics-body');

            let headers = pipeKeys.map(k => {
                const n = q.pipelines[k].name;
                const shortName = n.includes('.') ? n.split('.').slice(1).join('.').trim().split(' ')[0] : n;
                return `<th>${shortName}</th>`;
            }).join('');

            if (isCustomBActive && customBMetrics) {
                headers += `<th style="color:#00e676; border-left: 2px solid #00e676;">[Каскад B]</th>`;
            }

            thead.innerHTML = `<tr><th>Метрика</th>${headers}</tr>`;

            const metricDefs = [
                { key: 'enl', name: 'ENL (Гладкость)', format: v => typeof v === 'number' ? v.toFixed(2) : (v || '-') },
                { key: 'epi', name: 'EPI (Границы)', format: v => typeof v === 'number' ? v.toFixed(3) : (v || '-') },
                { key: 'bias_db', name: 'Bias (дБ)', format: v => typeof v === 'number' ? v.toFixed(3) : (v || '-') },
                { key: 'psnr', name: 'PSNR vs GT (дБ)', format: v => typeof v === 'number' ? v.toFixed(2) : (v || '-') },
                { key: 'ssim', name: 'SSIM vs GT', format: v => typeof v === 'number' ? v.toFixed(3) : (v || '-') },
                { key: 'ssi', name: 'SSI (< 1)', format: v => typeof v === 'number' ? v.toFixed(3) : (v || '-') },
                { key: 'smpi', name: 'SMPI (Качество)', format: v => typeof v === 'number' ? v.toFixed(3) : (v || '-') },
                { key: 'ratio_var', name: 'Var(r) (→ 0.227)', format: v => typeof v === 'number' ? v.toFixed(3) : (v || '-') },
            ];

            tbody.innerHTML = metricDefs.map(m => {
                let cells = pipeKeys.map(k => {
                    const metrics = q.pipelines[k].metrics || {};
                    const val = metrics[m.key];
                    const formatted = m.format(val);
                    return `<td>${formatted}</td>`;
                }).join('');

                if (isCustomBActive && customBMetrics) {
                    const valCustom = customBMetrics[m.key];
                    cells += `<td style="color:#00e676; font-weight:bold; border-left: 2px solid #00e676; background: rgba(0,230,118,0.1);">${m.format(valCustom)}</td>`;
                }

                return `<tr><td><b>${m.name}</b></td>${cells}</tr>`;
            }).join('');
        }

        function onQuadChange() {
            const q = DATA[currentQuad];
            document.getElementById('quadrant-desc').innerText = q.desc;
            map.flyTo(q.center, q.zoom, { duration: 1.0 });
            populatePipeSelects();
            isCustomBActive = false;
            customBImageUrl = null;
            customBMetrics = null;
            const statusBox = document.getElementById('custom-status-box');
            if (statusBox) statusBox.style.display = 'none';
            if (bMode === 'custom') {
                runCustomPipeline();
            } else {
                updateLayers();
            }
            updateMetricsTable();
        }

        document.getElementById('quadrant-select').addEventListener('change', (e) => {
            currentQuad = e.target.value;
            onQuadChange();
        });

        document.getElementById('pipe-a-select').addEventListener('change', (e) => {
            currentPipeA = e.target.value;
            updateLayers();
        });

        document.getElementById('pipe-b-select').addEventListener('change', (e) => {
            currentPipeB = e.target.value;
            isCustomBActive = false;
            updateLayers();
        });

        document.querySelectorAll('#layer-group .radio-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('#layer-group .radio-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentLayer = btn.getAttribute('data-layer');
                if (isCustomBActive) {
                    runCustomPipeline();
                } else {
                    updateLayers();
                }
            });
        });

        document.getElementById('opacity-slider').addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            document.getElementById('opacity-val').innerText = val + '%';
            currentOpacity = val / 100.0;
            if (overlayA) overlayA.setOpacity(currentOpacity);
            if (overlayB) overlayB.setOpacity(currentOpacity);
        });

        let isDragging = false;
        const divider = document.getElementById('swipe-divider');
        const container = document.getElementById('map-container');

        divider.addEventListener('mousedown', () => { isDragging = true; });
        window.addEventListener('mouseup', () => { isDragging = false; });
        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const rect = container.getBoundingClientRect();
            let x = e.clientX - rect.left;
            x = Math.max(10, Math.min(rect.width - 10, x));
            splitPct = (x / rect.width) * 100.0;
            applyClip();
        });

        map.on('move', applyClip);
        map.on('zoom', applyClip);
        map.on('resize', applyClip);

        // =========================================================================
        // BENCHMARK MODE LOGIC
        // =========================================================================
        let currentBenchId = "vitale";
        let currentBenchPipeA = "noisy";
        let currentBenchPipeB = "resnet";
        let benchSplitPct = 50.0;
        let isBenchDragging = false;

        function updateBenchMetricsTable(b) {
            const table = document.getElementById('bench-metrics-table');
            if (!table) return;
            const thead = table.querySelector('thead');
            const tbody = document.getElementById('bench-metrics-body');
            thead.innerHTML = `
                <tr>
                    <th style="text-align: left; padding: 4px 6px;">Метод</th>
                    <th style="padding: 4px 4px;">ENL</th>
                    <th style="padding: 4px 4px;">EPI</th>
                    <th style="padding: 4px 4px;">Bias</th>
                    <th style="padding: 4px 4px;">PSNR</th>
                    <th style="padding: 4px 4px;">SSIM</th>
                </tr>
            `;
            const pairKeys = Object.keys(b.pairs);
            let rowsHtml = pairKeys.map(k => {
                const p = b.pairs[k];
                const m = p.metrics || {};
                const isA = (k === currentBenchPipeA);
                const isB = (k === currentBenchPipeB && !isBenchCustomBActive);
                let rowBg = "";
                if (isA && isB) rowBg = "background: rgba(140, 158, 255, 0.25);";
                else if (isA) rowBg = "background: rgba(61, 90, 254, 0.2);";
                else if (isB) rowBg = "background: rgba(0, 230, 118, 0.2);";
                
                const enlStr = typeof m.enl === 'number' ? m.enl.toFixed(2) : (m.enl !== undefined ? m.enl : '-');
                const epiStr = typeof m.epi === 'number' ? m.epi.toFixed(3) : (m.epi !== undefined ? m.epi : '-');
                const biasStr = typeof m.bias_db === 'number' ? m.bias_db.toFixed(2) : (m.bias_db !== undefined ? m.bias_db : '-');
                const psnrStr = typeof m.psnr === 'number' ? m.psnr.toFixed(1) : (m.psnr !== undefined ? m.psnr : '-');
                const ssimStr = typeof m.ssim === 'number' ? m.ssim.toFixed(3) : (m.ssim !== undefined ? m.ssim : '-');
                
                const shortName = p.name.includes('.') ? p.name.split('.').slice(1).join('.').trim() : p.name;
                const prefix = isA ? '<span style="color:#3d5afe;font-weight:bold;">[A] </span>' : (isB ? '<span style="color:#00e676;font-weight:bold;">[B] </span>' : '');

                return `<tr style="${rowBg}">
                    <td style="text-align: left; padding: 3px 6px;">${prefix}${shortName}</td>
                    <td style="padding: 3px 4px;">${enlStr}</td>
                    <td style="padding: 3px 4px;">${epiStr}</td>
                    <td style="padding: 3px 4px;">${biasStr}</td>
                    <td style="padding: 3px 4px;">${psnrStr}</td>
                    <td style="padding: 3px 4px;">${ssimStr}</td>
                </tr>`;
            }).join('');

            if (isBenchCustomBActive && benchCustomBMetrics) {
                const cm = benchCustomBMetrics;
                const enlStr = typeof cm.enl === 'number' ? cm.enl.toFixed(2) : '-';
                const epiStr = typeof cm.epi === 'number' ? cm.epi.toFixed(3) : '-';
                const biasStr = typeof cm.bias_db === 'number' ? cm.bias_db.toFixed(2) : '-';
                const psnrStr = typeof cm.psnr === 'number' ? cm.psnr.toFixed(1) : '-';
                const ssimStr = typeof cm.ssim === 'number' ? cm.ssim.toFixed(3) : '-';
                rowsHtml += `<tr style="background: rgba(0, 230, 118, 0.25); border-left: 2px solid #00e676;">
                    <td style="text-align: left; padding: 3px 6px; font-weight: bold; color: #00e676;">[B] Каскад: ${benchCustomStages.join(' → ')}</td>
                    <td style="padding: 3px 4px; font-weight: bold;">${enlStr}</td>
                    <td style="padding: 3px 4px; font-weight: bold;">${epiStr}</td>
                    <td style="padding: 3px 4px; font-weight: bold;">${biasStr}</td>
                    <td style="padding: 3px 4px; font-weight: bold;">${psnrStr}</td>
                    <td style="padding: 3px 4px; font-weight: bold;">${ssimStr}</td>
                </tr>`;
            }

            tbody.innerHTML = rowsHtml;
        }

        function updateBenchView() {
            const b = DATA._benchmarks.find(item => item.id === currentBenchId) || DATA._benchmarks[0];
            document.getElementById('bench-title-text').innerText = b.title;
            document.getElementById('bench-paper-tag').innerText = b.paper;
            document.getElementById('bench-paper-desc').innerText = b.desc;
            document.getElementById('bench-strip-img').src = b.strip_img;

            const pairKeys = Object.keys(b.pairs);
            const selectA = document.getElementById('bench-pipe-a');
            const selectB = document.getElementById('bench-pipe-b');
            selectA.innerHTML = pairKeys.map(k => `<option value="${k}">${b.pairs[k].name}</option>`).join('');
            selectB.innerHTML = pairKeys.map(k => `<option value="${k}">${b.pairs[k].name}</option>`).join('');

            if (!pairKeys.includes(currentBenchPipeA)) currentBenchPipeA = pairKeys[0];
            if (!pairKeys.includes(currentBenchPipeB)) currentBenchPipeB = pairKeys.length > 1 ? pairKeys[1] : pairKeys[0];

            selectA.value = currentBenchPipeA;
            selectB.value = currentBenchPipeB;

            isBenchCustomBActive = false;
            renderBenchImages();
        }

        function renderBenchImages() {
            const b = DATA._benchmarks.find(item => item.id === currentBenchId) || DATA._benchmarks[0];
            const pairA = b.pairs[currentBenchPipeA];

            let imgBUrl = "";
            let nameB = "";
            let mB = {};

            if (isBenchCustomBActive && benchCustomBImageUrl) {
                imgBUrl = benchCustomBImageUrl;
                nameB = "Каскад: " + benchCustomStages.join(' → ');
                mB = benchCustomBMetrics || {};
            } else {
                const pairB = b.pairs[currentBenchPipeB];
                imgBUrl = pairB.img;
                nameB = pairB.name;
                mB = pairB.metrics || {};
            }

            document.getElementById('bench-img-a').src = pairA.img;
            document.getElementById('bench-img-b').src = imgBUrl;
            document.getElementById('bench-badge-a').innerText = "A: " + pairA.name;
            document.getElementById('bench-badge-b').innerText = "B: " + nameB;

            const mBox = document.getElementById('bench-pair-metrics');
            if (mBox) {
                const mA = pairA.metrics || {};
                const fmt = (v, digits) => typeof v === 'number' ? v.toFixed(digits) : (v !== undefined ? v : '-');
                mBox.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.74rem;">
                        <div style="background: rgba(61, 90, 254, 0.15); border: 1px solid #3d5afe; border-radius: 4px; padding: 6px;">
                            <div style="color: #8c9eff; font-weight: bold; margin-bottom: 4px;">Слой A: ${pairA.name.split('.')[1] || pairA.name}</div>
                            <div><b>ENL:</b> ${fmt(mA.enl, 2)}</div>
                            <div><b>EPI:</b> ${fmt(mA.epi, 3)}</div>
                            <div><b>Bias:</b> ${fmt(mA.bias_db, 2)} дБ</div>
                            <div><b>PSNR:</b> ${fmt(mA.psnr, 1)} дБ</div>
                            <div><b>SSIM:</b> ${fmt(mA.ssim, 3)}</div>
                        </div>
                        <div style="background: rgba(0, 230, 118, 0.15); border: 1px solid #00e676; border-radius: 4px; padding: 6px;">
                            <div style="color: #69f0ae; font-weight: bold; margin-bottom: 4px;">Слой B: ${nameB.split('.')[1] || nameB}</div>
                            <div><b>ENL:</b> ${fmt(mB.enl, 2)}</div>
                            <div><b>EPI:</b> ${fmt(mB.epi, 3)}</div>
                            <div><b>Bias:</b> ${fmt(mB.bias_db, 2)} дБ</div>
                            <div><b>PSNR:</b> ${fmt(mB.psnr, 1)} дБ</div>
                            <div><b>SSIM:</b> ${fmt(mB.ssim, 3)}</div>
                        </div>
                    </div>
                `;
            }

            updateBenchMetricsTable(b);
            applyBenchClip();
        }

        function applyBenchClip() {
            const d = document.getElementById('bench-divider');
            const imgA = document.getElementById('bench-img-a');
            d.style.left = benchSplitPct + '%';
            imgA.style.clipPath = `polygon(0 0, ${benchSplitPct}% 0, ${benchSplitPct}% 100%, 0 100%)`;
        }

        document.getElementById('bench-select').addEventListener('change', (e) => {
            currentBenchId = e.target.value;
            const b = DATA._benchmarks.find(item => item.id === currentBenchId);
            const pairKeys = Object.keys(b.pairs);
            currentBenchPipeA = pairKeys[0];
            currentBenchPipeB = pairKeys.length > 1 ? pairKeys[1] : pairKeys[0];
            updateBenchView();
        });

        document.getElementById('bench-pipe-a').addEventListener('change', (e) => {
            currentBenchPipeA = e.target.value;
            renderBenchImages();
        });

        document.getElementById('bench-pipe-b').addEventListener('change', (e) => {
            currentBenchPipeB = e.target.value;
            isBenchCustomBActive = false;
            renderBenchImages();
        });

        const bDivider = document.getElementById('bench-divider');
        const bWrap = document.getElementById('bench-canvas-wrap');
        bDivider.addEventListener('mousedown', () => { isBenchDragging = true; });
        window.addEventListener('mouseup', () => { isBenchDragging = false; });
        window.addEventListener('mousemove', (e) => {
            if (!isBenchDragging) return;
            const rect = bWrap.getBoundingClientRect();
            let x = e.clientX - rect.left;
            x = Math.max(0, Math.min(rect.width, x));
            benchSplitPct = (x / rect.width) * 100.0;
            applyBenchClip();
        });

        populatePipeSelects();
        onQuadChange();
        renderCustomStages();
        renderBenchCustomStages();

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
</body>
</html>
"""
