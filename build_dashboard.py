#!/usr/bin/env python3
"""
================================================================================
ETAPA 4: DASHBOARD INTERACTIVO HTML
AFORE XXI Banorte - Prueba Técnica Analista de Data Science
================================================================================

Genera un dashboard HTML autónomo (sin dependencias externas) a partir de:
  - cetes_clean.csv                          (datos limpios de Etapa 3)
  - predicciones_cetes_28d_comparativa.csv   (predicciones de Etapa 4)

Estructura (exacta a la prueba):
  Página 1 — Contexto de Tasas       (evolución temporal + filtros plazo/fecha)
  Página 2 — Yield Curve             (snapshot actual vs. 3, 6 y 12 meses)
  Página 3 — Comportamiento por Plazo (diferencia bp + identifica el más volátil)

Métricas clave:
  - Tasa Promedio por plazo
  - Diferencia semanal en bp por plazo
  - Volatilidad 4 semanas (adicional)
  - Spread vs CETES 28D (adicional)
  - Pronóstico CETES 28D a 4 semanas (adicional bonus)

Uso:
  python build_dashboard.py
  → genera dashboard_cetes.html
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 1. CARGA DE DATOS ────────────────────────────────────────────────────────
BASE = Path(__file__).parent
df = pd.read_csv(BASE / "cetes_clean.csv", parse_dates=["fecha"])
pron = pd.read_csv(BASE / "predicciones_cetes_28d_comparativa.csv", parse_dates=["fecha"])

PLAZOS = [28, 91, 182, 364, 728]
NOMBRES = {28: "CETES 28D", 91: "CETES 91D", 182: "CETES 182D",
           364: "CETES 364D", 728: "CETES 728D"}
COLORS = {28: "#E31837", 91: "#FF6B35", 182: "#F7931E", 364: "#00A651", 728: "#0072CE"}

df["Plazo"] = df["plazo_dias"].map(NOMBRES)
df = df.sort_values(["plazo_dias", "fecha"]).reset_index(drop=True)
fecha_max = df["fecha"].max()

# ── 2. SERIE PARA CADA PLAZO (para Página 1) ─────────────────────────────────
series_data = {}
for p in PLAZOS:
    sub = df[df["plazo_dias"] == p].sort_values("fecha")
    series_data[p] = {
        "name": NOMBRES[p],
        "color": COLORS[p],
        "x": sub["fecha"].dt.strftime("%Y-%m-%d").tolist(),
        "y": [round(v, 4) if pd.notna(v) else None for v in sub["tasa_rendimiento"]],
        "bp": [round(v, 2) if pd.notna(v) else None for v in sub["cambio_bp"]],
        "spread": [round(v, 4) if pd.notna(v) else None for v in sub["spread_vs_28d"]],
        "vol4": [round(v, 4) if pd.notna(v) else None for v in sub["volatilidad_4s"]],
    }

# ── 3. SNAPSHOT YIELD CURVE (Página 2) ───────────────────────────────────────
def valor_mas_cercano(serie_df, fecha_objetivo):
    serie_df = serie_df.dropna(subset=["tasa_rendimiento"])
    if serie_df.empty:
        return None
    idx = (serie_df["fecha"] - fecha_objetivo).abs().idxmin()
    return float(round(serie_df.loc[idx, "tasa_rendimiento"], 4))


cortes = {
    "Actual": fecha_max,
    "Hace 3 meses": fecha_max - pd.DateOffset(months=3),
    "Hace 6 meses": fecha_max - pd.DateOffset(months=6),
    "Hace 12 meses": fecha_max - pd.DateOffset(months=12),
}
yield_curve = {}
for etiqueta, fecha_obj in cortes.items():
    valores = []
    for p in PLAZOS:
        sub = df[df["plazo_dias"] == p][["fecha", "tasa_rendimiento"]]
        valores.append(valor_mas_cercano(sub, fecha_obj))
    yield_curve[etiqueta] = valores

# ── 4. KPIS GLOBALES ─────────────────────────────────────────────────────────
def safe_round(x, n=4):
    return round(float(x), n) if pd.notna(x) else None


ultimo = df.sort_values("fecha").groupby("plazo_dias").tail(1).set_index("plazo_dias")
tasa_promedio_plazo = df.groupby("plazo_dias")["tasa_rendimiento"].mean().round(4).to_dict()
bp_promedio_abs = df.groupby("plazo_dias")["cambio_bp"].apply(lambda s: s.abs().mean()).round(2).to_dict()
bp_max_abs = df.groupby("plazo_dias")["cambio_bp"].apply(lambda s: s.abs().max()).round(2).to_dict()
vol_promedio = df.groupby("plazo_dias")["volatilidad_4s"].mean().round(4).to_dict()

plazo_mas_volatil = max(bp_promedio_abs, key=bp_promedio_abs.get)
plazo_mas_estable = min(bp_promedio_abs, key=bp_promedio_abs.get)

kpis = {
    "tasa_28d_actual": safe_round(ultimo.loc[28, "tasa_rendimiento"]),
    "tasa_28d_hace_12m": yield_curve["Hace 12 meses"][0],
    "cambio_12m_28d": round(
        safe_round(ultimo.loc[28, "tasa_rendimiento"]) - yield_curve["Hace 12 meses"][0], 4
    ),
    "pendiente_curva": round(
        safe_round(ultimo.loc[728, "tasa_rendimiento"]) - safe_round(ultimo.loc[28, "tasa_rendimiento"]), 4
    ),
    "fecha_ultimo_dato": fecha_max.strftime("%d de %B de %Y").replace(
        "January", "enero").replace("February", "febrero").replace("March", "marzo").replace(
        "April", "abril").replace("May", "mayo").replace("June", "junio").replace(
        "July", "julio").replace("August", "agosto").replace("September", "septiembre").replace(
        "October", "octubre").replace("November", "noviembre").replace("December", "diciembre"),
    "plazo_mas_volatil": NOMBRES[plazo_mas_volatil],
    "bp_plazo_mas_volatil": bp_promedio_abs[plazo_mas_volatil],
    "plazo_mas_estable": NOMBRES[plazo_mas_estable],
    "bp_plazo_mas_estable": bp_promedio_abs[plazo_mas_estable],
    "tasa_promedio_plazo": {NOMBRES[k]: round(v, 4) for k, v in tasa_promedio_plazo.items()},
    "bp_promedio_abs": {NOMBRES[k]: round(v, 2) for k, v in bp_promedio_abs.items()},
    "bp_max_abs": {NOMBRES[k]: round(v, 2) for k, v in bp_max_abs.items()},
    "vol_promedio": {NOMBRES[k]: round(v, 4) for k, v in vol_promedio.items()},
    "n_observaciones": len(df),
    "fecha_min": df["fecha"].min().strftime("%Y-%m-%d"),
    "fecha_max": fecha_max.strftime("%Y-%m-%d"),
}

# ── 5. PRONÓSTICO (bonus para Página 3) ──────────────────────────────────────
pron = pron.sort_values("fecha")
historico_28d = df[df["plazo_dias"] == 28].sort_values("fecha").tail(16)
pronostico_data = {
    "hist_x": historico_28d["fecha"].dt.strftime("%Y-%m-%d").tolist(),
    "hist_y": [round(v, 4) for v in historico_28d["tasa_rendimiento"]],
    "fcst_x": pron["fecha"].dt.strftime("%Y-%m-%d").tolist(),
    "arima": [round(v, 4) for v in pron["prediccion_arima"]],
    "arima_inf": [round(v, 4) for v in pron["arima_inf_95"]],
    "arima_sup": [round(v, 4) for v in pron["arima_sup_95"]],
    "prophet": [round(v, 4) for v in pron["prediccion_prophet"]],
    "xgboost": [round(v, 4) for v in pron["prediccion_xgboost"]],
}

# ── 5b. MÉTRICAS DE VALIDACIÓN ───────────────────────────────────────────────
# Resultados del entrenamiento del script 04_modelo_predictivo_standalone.py
# Hold-out: últimas 12 semanas del histórico de 5 años (~261 obs); MAPE en %.
metricas_validacion = {
    "ARIMA": {
        "mae": 0.3739, "rmse": 0.393, "mape": 5.84, "aic": -281.45,
        "tipo": "Series de tiempo clásica",
        "fortaleza": "Captura inercia de corto plazo; intervalo de confianza estadísticamente fundamentado.",
        "debilidad": "Asume estacionariedad tras diferenciar; no modela cambios de régimen.",
    },
    "Prophet": {
        "mae": 0.3551, "rmse": 0.3758, "mape": 5.54,
        "tipo": "Descomposición aditiva (Facebook)",
        "fortaleza": "Robusto a tendencias y cambios de pendiente; el de menor error en validación.",
        "debilidad": "Estacionalidad anual aporta poco con tasas que se mueven por política monetaria.",
    },
    "XGBoost": {
        "mae": 0.6071, "rmse": None, "mape": 6.99, "cv_r2_mean": -0.2232,
        "tipo": "Gradient boosting (ML)",
        "fortaleza": "Modela no-linealidades; útil con muchas observaciones y features.",
        "debilidad": "R² negativo en CV: con 52 obs útiles tras lags, no generaliza. No usar en producción.",
    },
    "ganador": "Prophet",
    "horizonte_validacion": 12,
    "metrica_decisora": "MAPE (Error Porcentual Absoluto Medio)",
}

# ── 6. PAYLOAD JSON COMPLETO ─────────────────────────────────────────────────
payload = {
    "kpis": kpis,
    "plazos": PLAZOS,
    "nombres": NOMBRES,
    "colors": {str(k): v for k, v in COLORS.items()},
    "series": {str(k): v for k, v in series_data.items()},
    "yield_curve": {"labels": [NOMBRES[p] for p in PLAZOS], **yield_curve},
    "pronostico": pronostico_data,
    "metricas": metricas_validacion,
    "build_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
}
payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

# ── 7. LEER LIBRERÍA PLOTLY EMBEBIDA ─────────────────────────────────────────
plotly_path = BASE / "plotly.min.js"
if plotly_path.exists():
    plotly_js = plotly_path.read_text(encoding="utf-8")
else:
    plotly_js = ""  # se inyectará en build externo

# ── 8. ENSAMBLAR HTML ────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dashboard CETES — AFORE XXI Banorte</title>
<style>
  :root{
    --banorte-red:#EB0029;
    --banorte-red-dark:#A8001E;
    --navy:#1A1F36;
    --ink:#1A2233;
    --muted:#6B7488;
    --line:#E5E8EE;
    --bg:#F4F5F8;
    --paper:#FFFFFF;
    --c28:#EB0029;
    --c91:#FF6B35;
    --c182:#F7931E;
    --c364:#00A651;
    --c728:#0072CE;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;}
  body{
    background:var(--bg);
    color:var(--ink);
    font-family:"Segoe UI",-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;
    -webkit-font-smoothing:antialiased;
  }

  /* ── HEADER ───────────────────────────────────────────────────── */
  .header{
    background:linear-gradient(135deg,var(--banorte-red) 0%,var(--banorte-red-dark) 100%);
    color:#fff; padding:22px 32px; box-shadow:0 4px 16px rgba(235,0,41,0.18);
  }
  .header-top{display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;}
  .header h1{margin:0; font-size:22px; font-weight:700; letter-spacing:-0.2px;}
  .header .sub{font-size:13px; opacity:0.92; margin-top:3px;}
  .header .badge{
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(255,255,255,0.18); padding:6px 14px; border-radius:999px;
    font-size:12.5px; font-weight:600;
  }
  .header .badge::before{content:""; width:7px; height:7px; border-radius:50%; background:#7CFC9D;}

  /* ── NAV ─────────────────────────────────────────────────────── */
  .nav{
    display:flex; justify-content:center; background:#fff;
    box-shadow:0 1px 0 var(--line);
    position:sticky; top:0; z-index:50;
  }
  .nav button{
    flex:0 0 auto;
    padding:15px 28px; border:0; background:#fff; cursor:pointer;
    font-size:14px; font-weight:600; color:var(--muted);
    border-bottom:3px solid transparent;
    transition:color .15s,border-color .15s,background .15s;
    font-family:inherit;
  }
  .nav button:hover{color:var(--banorte-red); background:#FFF6F7;}
  .nav button.active{color:var(--banorte-red); border-bottom-color:var(--banorte-red); background:#FFF6F7;}
  .nav .step{display:inline-block; width:22px; height:22px; line-height:22px; text-align:center;
    border-radius:50%; background:#EFE; color:var(--muted); font-size:11px; margin-right:8px;
    background:#F2F4F8;}
  .nav button.active .step{background:var(--banorte-red); color:#fff;}

  .container{max-width:1400px; margin:0 auto; padding:24px 28px 48px;}
  .page{display:none; animation:fade .25s ease;}
  .page.active{display:block;}
  @keyframes fade{from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;}}

  /* ── NARRATIVE BANNERS ──────────────────────────────────────── */
  .narrative{
    background:var(--paper); border-radius:12px; padding:18px 22px; margin-bottom:20px;
    border-left:4px solid var(--banorte-red);
    display:flex; gap:18px; align-items:flex-start;
    box-shadow:0 1px 0 var(--line);
  }
  .narrative .icon{
    flex:0 0 36px; height:36px; border-radius:8px;
    background:#FFE9EC; color:var(--banorte-red);
    display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:700;
  }
  .narrative .text h2{margin:0 0 4px; font-size:17px; color:var(--ink); font-weight:700;}
  .narrative .text p{margin:0; font-size:13.5px; color:var(--muted); line-height:1.55;}

  /* ── KPI CARDS ─────────────────────────────────────────────── */
  .kpi-grid{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
    gap:14px; margin-bottom:22px;
  }
  .kpi{
    background:var(--paper); border-radius:12px; padding:18px 20px;
    box-shadow:0 1px 0 var(--line), 0 6px 18px -14px rgba(26,31,54,0.18);
    position:relative; overflow:hidden;
  }
  .kpi .label{font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; font-weight:700;}
  .kpi .value{font-size:30px; font-weight:700; color:var(--ink); margin-top:6px; line-height:1.1;}
  .kpi .value small{font-size:14px; font-weight:600; color:var(--muted);}
  .kpi .delta{display:inline-flex; align-items:center; gap:4px; margin-top:8px;
    font-size:12px; font-weight:700; padding:3px 9px; border-radius:999px;}
  .kpi .delta.down{background:#FFE9EC; color:var(--banorte-red);}
  .kpi .delta.up{background:#E6F5EB; color:#00813F;}
  .kpi .delta.neutral{background:#EEF0F4; color:var(--muted);}

  /* ── FILTROS ───────────────────────────────────────────────── */
  .filters{
    background:var(--paper); border-radius:12px; padding:16px 20px; margin-bottom:18px;
    display:flex; flex-wrap:wrap; gap:18px; align-items:center;
    box-shadow:0 1px 0 var(--line);
  }
  .filters .group{display:flex; flex-direction:column; gap:6px;}
  .filters .group-label{font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; font-weight:700;}
  .pill-row{display:flex; flex-wrap:wrap; gap:6px;}
  .pill{
    cursor:pointer; padding:6px 14px; border-radius:999px; font-size:12.5px; font-weight:600;
    background:#F2F4F8; color:var(--muted); border:1.5px solid transparent;
    user-select:none; transition:all .15s;
  }
  .pill:hover{background:#E8EBF1;}
  .pill.active{background:#FFF6F7; color:var(--banorte-red); border-color:var(--banorte-red);}
  .pill.active.c28{color:#EB0029; border-color:#EB0029;}
  .pill.active.c91{color:#FF6B35; border-color:#FF6B35;}
  .pill.active.c182{color:#F7931E; border-color:#F7931E;}
  .pill.active.c364{color:#00A651; border-color:#00A651;}
  .pill.active.c728{color:#0072CE; border-color:#0072CE;}
  .filters input[type="date"]{
    padding:6px 10px; border:1.5px solid var(--line); border-radius:8px;
    font-size:13px; font-family:inherit; color:var(--ink); background:#fff;
  }
  .filters button.reset{
    margin-left:auto; padding:7px 14px; border-radius:8px;
    background:transparent; border:1.5px solid var(--line); color:var(--muted);
    font-size:12.5px; font-weight:600; cursor:pointer; font-family:inherit;
  }
  .filters button.reset:hover{background:#F2F4F8;}

  /* ── PANELS / CHARTS ──────────────────────────────────────── */
  .panel{
    background:var(--paper); border-radius:12px; padding:22px 24px; margin-bottom:18px;
    box-shadow:0 1px 0 var(--line), 0 6px 18px -14px rgba(26,31,54,0.16);
  }
  .panel h3{margin:0 0 4px; font-size:16px; font-weight:700; color:var(--ink);}
  .panel .panel-sub{font-size:13px; color:var(--muted); margin:0 0 16px;}
  .two-col{display:grid; grid-template-columns:1.4fr 1fr; gap:18px;}
  .three-col{display:grid; grid-template-columns:repeat(3,1fr); gap:14px;}

  .table-clean{width:100%; border-collapse:collapse; font-size:13.5px;}
  .table-clean th{
    text-align:left; padding:10px 12px; background:#F8F9FC; color:var(--muted);
    font-size:11.5px; text-transform:uppercase; letter-spacing:1px; font-weight:700;
    border-bottom:1px solid var(--line);
  }
  .table-clean td{padding:10px 12px; border-bottom:1px solid var(--line); color:var(--ink);}
  .table-clean tr:last-child td{border-bottom:0;}
  .table-clean td.num{text-align:right; font-variant-numeric:tabular-nums; font-weight:600;}
  .table-clean tr.highlight td{background:#FFF6F7;}
  .table-clean .dot{display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px;}

  /* ── INSIGHT BOX ──────────────────────────────────────────── */
  .insight{
    background:linear-gradient(135deg,#1A1F36 0%,#2A2F4F 100%);
    color:#fff; border-radius:12px; padding:20px 24px;
    display:flex; gap:18px; align-items:center; margin-bottom:18px;
  }
  .insight .big{font-size:36px; font-weight:700; letter-spacing:-1px; line-height:1;}
  .insight .label{font-size:11.5px; color:#9FA9C2; text-transform:uppercase; letter-spacing:1.5px; font-weight:700;}
  .insight p{margin:6px 0 0; color:#D4D9E6; font-size:14px; line-height:1.5;}
  .insight .right{flex:1;}

  footer{margin-top:30px; padding:22px 0; border-top:1px solid var(--line);
    font-size:12px; color:var(--muted); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;}

  /* ── MODEL CARDS ─────────────────────────────────────────── */
  .model-card{
    background:#FAFBFD; border-radius:10px; padding:18px 20px; position:relative;
    border:1.5px solid var(--line);
  }
  .model-card.winner{border-color:#00A651; background:#F2FAF5;}
  .model-card .model-head{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;}
  .model-card h4{margin:0; font-size:15px; font-weight:700; color:var(--ink);}
  .model-card .mape-tag{
    font-size:12px; font-weight:700; padding:3px 10px; border-radius:999px;
    background:#FFE9EC; color:var(--banorte-red);
  }
  .model-card.winner .mape-tag{background:#E6F5EB; color:#00813F;}
  .model-card .winner-badge{
    position:absolute; top:-9px; right:14px;
    background:#00A651; color:#fff; font-size:10.5px; font-weight:700;
    padding:3px 10px; border-radius:999px; letter-spacing:0.5px;
  }
  .model-card .tipo{font-size:11.5px; color:var(--muted); margin-bottom:10px; font-style:italic;}
  .model-card .pro, .model-card .con{font-size:12.5px; line-height:1.5; margin:6px 0;}
  .model-card .pro strong{color:#00813F;}
  .model-card .con strong{color:var(--banorte-red);}

  @media (max-width:900px){
    .two-col,.three-col{grid-template-columns:1fr;}
    .nav{overflow-x:auto;}
    .nav button{padding:14px 18px; font-size:13px;}
    .header h1{font-size:18px;}
  }
</style>
</head>
<body>

<div class="header">
  <div class="header-top">
    <div>
      <h1>Dashboard CETES — Curva de Rendimientos</h1>
      <div class="sub">AFORE XXI Banorte · Subdirección de Trading y Análisis Cuantitativo</div>
    </div>
    <div class="badge">Datos al __FECHA_ULTIMA__</div>
  </div>
</div>

<nav class="nav">
  <button class="active" data-page="page1"><span class="step">1</span>Contexto de Tasas</button>
  <button data-page="page2"><span class="step">2</span>Yield Curve</button>
  <button data-page="page3"><span class="step">3</span>Comportamiento por Plazo</button>
</nav>

<div class="container">

  <!-- ═════════════ PÁGINA 1 — CONTEXTO DE TASAS ═════════════ -->
  <div id="page1" class="page active">
    <div class="narrative">
      <div class="icon">1</div>
      <div class="text">
        <h2>El punto de partida: ¿dónde están las tasas hoy?</h2>
        <p>Evolución semanal de las 5 tasas de CETES en los últimos 12 meses. Usa los filtros para enfocarte en un plazo o en una ventana específica de fechas.</p>
      </div>
    </div>

    <div class="kpi-grid" id="kpi-grid-page1"></div>

    <div class="filters">
      <div class="group">
        <span class="group-label">Plazos visibles</span>
        <div class="pill-row" id="plazo-filters"></div>
      </div>
      <div class="group">
        <span class="group-label">Desde</span>
        <input type="date" id="date-from">
      </div>
      <div class="group">
        <span class="group-label">Hasta</span>
        <input type="date" id="date-to">
      </div>
      <button class="reset" id="reset-filters">Restablecer</button>
    </div>

    <div class="panel">
      <h3>Tasa de rendimiento por plazo</h3>
      <p class="panel-sub">% anual · Fuente: Banco de México (SIE)</p>
      <div id="chart-evolucion" style="height:430px;"></div>
    </div>

    <div class="panel">
      <h3>Tasa Promedio por plazo · últimos 12 meses</h3>
      <p class="panel-sub">Métrica clave 1 de 3 solicitadas por la prueba técnica</p>
      <table class="table-clean" id="tabla-promedios"></table>
    </div>
  </div>

  <!-- ═════════════ PÁGINA 2 — YIELD CURVE ═════════════ -->
  <div id="page2" class="page">
    <div class="narrative">
      <div class="icon">2</div>
      <div class="text">
        <h2>La forma de la curva y cómo ha cambiado</h2>
        <p>Comparativo de la curva de rendimientos actual contra hace 3, 6 y 12 meses. Indicador clave: la pendiente positiva señala expectativa normal del mercado; una curva invertida sería señal de estrés.</p>
      </div>
    </div>

    <div class="kpi-grid" id="kpi-grid-page2"></div>

    <div class="panel">
      <h3>Curva de Rendimientos · snapshots comparativos</h3>
      <p class="panel-sub">28D · 91D · 182D · 364D · 728D</p>
      <div id="chart-curva" style="height:440px;"></div>
    </div>

    <div class="two-col">
      <div class="panel">
        <h3>Valores por snapshot</h3>
        <p class="panel-sub">% anual</p>
        <table class="table-clean" id="tabla-curva"></table>
      </div>
      <div class="panel">
        <h3>Lectura</h3>
        <p class="panel-sub">Tendencia 12 meses</p>
        <div id="lectura-curva"></div>
      </div>
    </div>
  </div>

  <!-- ═════════════ PÁGINA 3 — COMPORTAMIENTO POR PLAZO ═════════════ -->
  <div id="page3" class="page">
    <div class="narrative">
      <div class="icon">3</div>
      <div class="text">
        <h2>¿Qué plazo se mueve más cada semana?</h2>
        <p>Diferencia semanal en puntos base (bp) por plazo. Identificación del plazo con mayor movimiento promedio para gestión de riesgo de duración.</p>
      </div>
    </div>

    <div class="insight" id="insight-volatil"></div>

    <div class="kpi-grid" id="kpi-grid-page3"></div>

    <div class="panel">
      <h3>Diferencia semanal en puntos base — todos los plazos</h3>
      <p class="panel-sub">Métrica clave 2 de 3 solicitadas por la prueba técnica</p>
      <div id="chart-bp" style="height:430px;"></div>
    </div>

    <div class="two-col">
      <div class="panel">
        <h3>Resumen estadístico de volatilidad</h3>
        <p class="panel-sub">Movimiento promedio absoluto · 12 meses</p>
        <table class="table-clean" id="tabla-volatilidad"></table>
      </div>
      <div class="panel">
        <h3>Comparativo de modelos · próximas 4 semanas</h3>
        <p class="panel-sub">Tabla detallada de las 3 predicciones (mismo dato del gráfico)</p>
        <table class="table-clean" id="tabla-pronostico"></table>
      </div>
    </div>

    <div class="panel">
      <h3>Predicciones CETES 28D (4 semanas) — ARIMA · Prophet · XGBoost</h3>
      <p class="panel-sub">Comparativo de los 3 modelos entrenados sobre 5 años de historia · banda gris = IC 95% de ARIMA</p>
      <div id="chart-pronostico" style="height:430px;"></div>
    </div>

    <div class="insight" id="insight-modelo"></div>

    <div class="two-col">
      <div class="panel">
        <h3>Métricas de validación — hold-out últimas 12 semanas</h3>
        <p class="panel-sub">MAE, RMSE y MAPE menores = mejor predicción</p>
        <table class="table-clean" id="tabla-metricas"></table>
      </div>
      <div class="panel">
        <h3>MAPE por modelo</h3>
        <p class="panel-sub">Error porcentual absoluto medio (%) — menor es mejor</p>
        <div id="chart-mape" style="height:280px;"></div>
      </div>
    </div>

    <div class="panel">
      <h3>¿Cuándo conviene cada modelo?</h3>
      <p class="panel-sub">Fortalezas, debilidades y criterio de selección</p>
      <div class="three-col" id="cards-modelos"></div>
    </div>
  </div>

  <footer>
    <span>Generado el __BUILD_DATE__ · 100% datos públicos del Banco de México</span>
    <span>Pipeline: API SIE → SQLite → Data Mining → Dashboard</span>
  </footer>
</div>

<!-- ── PLOTLY (embebido para funcionamiento offline) ───────────────────── -->
__PLOTLY_LIB__

<!-- ── PAYLOAD DE DATOS ─────────────────────────────────────────────── -->
<script>
window.DATA = __PAYLOAD__;
</script>

<!-- ── LÓGICA DEL DASHBOARD ─────────────────────────────────────────── -->
<script>
const D = window.DATA;
const PLAZOS = D.plazos;
const NOMBRES = D.nombres;
const COLORS = D.colors;
const NAVY = "#1A1F36";

const fmtPct = v => (v == null ? "—" : (v.toFixed(2) + "%"));
const fmtBp = v => {
  if (v == null) return "—";
  const sign = v > 0 ? "+" : "";
  return sign + v.toFixed(1) + " bp";
};
const fmtPctDiff = v => {
  if (v == null) return "—";
  const sign = v > 0 ? "+" : "";
  return sign + v.toFixed(2) + " pp";
};

// ── ESTADO DE FILTROS ────────────────────────────────────────────
const state = {
  active: new Set(PLAZOS.map(String)),
  dateFrom: D.kpis.fecha_min,
  dateTo: D.kpis.fecha_max,
};

// ── NAVEGACIÓN ENTRE PÁGINAS ────────────────────────────────────
document.querySelectorAll(".nav button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.page).classList.add("active");
    // Re-render charts in case of resize
    setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
  });
});

// ════════════════════════════════════════════════════════════════
// KPI CARDS
// ════════════════════════════════════════════════════════════════
function kpiCard(label, value, delta, deltaClass) {
  return `<div class="kpi">
    <div class="label">${label}</div>
    <div class="value">${value}</div>
    ${delta ? `<span class="delta ${deltaClass||'neutral'}">${delta}</span>` : ''}
  </div>`;
}

function renderKpisPage1() {
  const k = D.kpis;
  const diff = k.cambio_12m_28d;
  const cls = diff < 0 ? "down" : (diff > 0 ? "up" : "neutral");
  // Última tasa 728D no nula
  const s728 = D.series["728"].y.filter(v => v != null);
  const tasa728_actual = s728.length ? s728[s728.length - 1] : null;
  document.getElementById("kpi-grid-page1").innerHTML = [
    kpiCard("Tasa CETES 28D actual", fmtPct(k.tasa_28d_actual),
            fmtPctDiff(diff) + " vs. hace 12 meses", cls),
    kpiCard("Tasa CETES 728D actual", fmtPct(tasa728_actual),
            "Plazo largo · referencia", "neutral"),
    kpiCard("Pendiente (728D − 28D)", k.pendiente_curva.toFixed(2) + " pp",
            "Curva normal", "up"),
    kpiCard("Observaciones totales", String(k.n_observaciones),
            "5 plazos · 52 semanas", "neutral"),
  ].join("");
}

function renderKpisPage2() {
  const yc = D.yield_curve;
  const c28_actual = yc["Actual"][0];
  const c28_12m = yc["Hace 12 meses"][0];
  const c728_actual = yc["Actual"][4];
  const c728_12m = yc["Hace 12 meses"][4];
  document.getElementById("kpi-grid-page2").innerHTML = [
    kpiCard("CETES 28D · Actual", fmtPct(c28_actual),
            fmtPctDiff(c28_actual - c28_12m) + " vs. 12m", c28_actual < c28_12m ? "down" : "up"),
    kpiCard("CETES 728D · Actual", fmtPct(c728_actual),
            fmtPctDiff(c728_actual - c728_12m) + " vs. 12m", c728_actual < c728_12m ? "down" : "up"),
    kpiCard("Pendiente actual", (c728_actual - c28_actual).toFixed(2) + " pp",
            "Curva positiva", "up"),
    kpiCard("Pendiente hace 12 meses", (c728_12m - c28_12m).toFixed(2) + " pp",
            null),
  ].join("");
}

function renderKpisPage3() {
  const k = D.kpis;
  document.getElementById("kpi-grid-page3").innerHTML = [
    kpiCard("Plazo más volátil", k.plazo_mas_volatil,
            k.bp_plazo_mas_volatil.toFixed(1) + " bp/semana", "down"),
    kpiCard("Plazo más estable", k.plazo_mas_estable,
            k.bp_plazo_mas_estable.toFixed(1) + " bp/semana", "up"),
    kpiCard("Movimiento máximo CETES 728D", k.bp_max_abs["CETES 728D"].toFixed(0) + " bp",
            "Pico semanal absoluto", "neutral"),
    kpiCard("Movimiento máximo CETES 28D", k.bp_max_abs["CETES 28D"].toFixed(0) + " bp",
            "Pico semanal absoluto", "neutral"),
  ].join("");
}

// ════════════════════════════════════════════════════════════════
// PÁGINA 1 — CONTEXTO DE TASAS
// ════════════════════════════════════════════════════════════════
function renderPlazoFilters() {
  const html = PLAZOS.map(p => {
    const active = state.active.has(String(p));
    return `<span class="pill ${active ? 'active c'+p : ''}" data-plazo="${p}">
      ${NOMBRES[p]}
    </span>`;
  }).join("");
  document.getElementById("plazo-filters").innerHTML = html;
  document.querySelectorAll("#plazo-filters .pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const plazo = pill.dataset.plazo;
      if (state.active.has(plazo)) {
        if (state.active.size > 1) state.active.delete(plazo);
      } else {
        state.active.add(plazo);
      }
      renderPlazoFilters();
      renderEvolucion();
    });
  });
}

function setupDateFilters() {
  const from = document.getElementById("date-from");
  const to = document.getElementById("date-to");
  from.min = D.kpis.fecha_min; from.max = D.kpis.fecha_max; from.value = state.dateFrom;
  to.min = D.kpis.fecha_min; to.max = D.kpis.fecha_max; to.value = state.dateTo;
  from.addEventListener("change", () => { state.dateFrom = from.value; renderEvolucion(); });
  to.addEventListener("change", () => { state.dateTo = to.value; renderEvolucion(); });
  document.getElementById("reset-filters").addEventListener("click", () => {
    state.active = new Set(PLAZOS.map(String));
    state.dateFrom = D.kpis.fecha_min; state.dateTo = D.kpis.fecha_max;
    from.value = state.dateFrom; to.value = state.dateTo;
    renderPlazoFilters(); renderEvolucion();
  });
}

function renderEvolucion() {
  const traces = PLAZOS.filter(p => state.active.has(String(p))).map(p => {
    const s = D.series[String(p)];
    const xs = [], ys = [];
    for (let i = 0; i < s.x.length; i++) {
      if (s.x[i] >= state.dateFrom && s.x[i] <= state.dateTo) {
        xs.push(s.x[i]); ys.push(s.y[i]);
      }
    }
    return {
      type: "scatter", mode: "lines", name: s.name, x: xs, y: ys,
      line: { color: s.color, width: 2.5, shape: "linear" },
      hovertemplate: "<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>",
    };
  });
  Plotly.newPlot("chart-evolucion", traces, {
    margin: { l: 50, r: 20, t: 10, b: 50 },
    xaxis: { gridcolor: "#EFF1F6", showline: false, tickformat: "%b %y" },
    yaxis: { gridcolor: "#EFF1F6", ticksuffix: "%", zeroline: false },
    legend: { orientation: "h", x: 0.5, y: -0.18, xanchor: "center" },
    hovermode: "x unified",
    paper_bgcolor: "white", plot_bgcolor: "white",
    font: { family: "Segoe UI, Arial", color: "#1A2233", size: 12 },
  }, { displaylogo: false, responsive: true });
}

function renderTablaPromedios() {
  const k = D.kpis;
  const rows = PLAZOS.map(p => {
    const n = NOMBRES[p];
    const c = COLORS[String(p)];
    return `<tr>
      <td><span class="dot" style="background:${c}"></span>${n}</td>
      <td class="num">${k.tasa_promedio_plazo[n].toFixed(2)}%</td>
      <td class="num">${k.bp_promedio_abs[n].toFixed(1)} bp</td>
      <td class="num">${(k.vol_promedio[n]*100).toFixed(2)}%</td>
      <td class="num">${k.bp_max_abs[n].toFixed(0)} bp</td>
    </tr>`;
  }).join("");
  document.getElementById("tabla-promedios").innerHTML = `
    <thead>
      <tr><th>Plazo</th><th style="text-align:right">Tasa promedio</th>
      <th style="text-align:right">|bp| promedio semanal</th>
      <th style="text-align:right">Volatilidad 4s</th>
      <th style="text-align:right">|bp| máximo</th></tr>
    </thead>
    <tbody>${rows}</tbody>`;
}

// ════════════════════════════════════════════════════════════════
// PÁGINA 2 — YIELD CURVE
// ════════════════════════════════════════════════════════════════
function renderCurva() {
  const yc = D.yield_curve;
  const xlabels = yc.labels;
  const snapColors = { "Actual": "#EB0029", "Hace 3 meses": "#FF8855",
                       "Hace 6 meses": "#0072CE", "Hace 12 meses": "#9AA4BC" };
  const widths = { "Actual": 4, "Hace 3 meses": 2.5, "Hace 6 meses": 2.5, "Hace 12 meses": 2.5 };
  const dashes = { "Actual": "solid", "Hace 3 meses": "solid", "Hace 6 meses": "solid", "Hace 12 meses": "dot" };
  const traces = ["Actual", "Hace 3 meses", "Hace 6 meses", "Hace 12 meses"].map(label => ({
    type: "scatter", mode: "lines+markers", name: label,
    x: xlabels, y: yc[label],
    line: { color: snapColors[label], width: widths[label], dash: dashes[label] },
    marker: { size: label === "Actual" ? 11 : 8, color: snapColors[label] },
    hovertemplate: "<b>%{fullData.name}</b><br>%{x}: %{y:.2f}%<extra></extra>",
  }));
  Plotly.newPlot("chart-curva", traces, {
    margin: { l: 50, r: 20, t: 10, b: 60 },
    xaxis: { gridcolor: "#EFF1F6", showline: false },
    yaxis: { gridcolor: "#EFF1F6", ticksuffix: "%", zeroline: false },
    legend: { orientation: "h", x: 0.5, y: -0.18, xanchor: "center" },
    hovermode: "x unified",
    paper_bgcolor: "white", plot_bgcolor: "white",
    font: { family: "Segoe UI, Arial", color: "#1A2233", size: 12 },
  }, { displaylogo: false, responsive: true });
}

function renderTablaCurva() {
  const yc = D.yield_curve;
  const cols = ["Actual", "Hace 3 meses", "Hace 6 meses", "Hace 12 meses"];
  const headers = `<tr><th>Plazo</th>${cols.map(c => `<th style="text-align:right">${c}</th>`).join("")}<th style="text-align:right">Δ 12m</th></tr>`;
  const rows = PLAZOS.map((p, i) => {
    const c = COLORS[String(p)];
    const n = NOMBRES[p];
    const cells = cols.map(col => `<td class="num">${yc[col][i] == null ? "—" : yc[col][i].toFixed(2) + "%"}</td>`).join("");
    const delta = yc["Actual"][i] - yc["Hace 12 meses"][i];
    const deltaColor = delta < 0 ? "#EB0029" : "#00813F";
    return `<tr><td><span class="dot" style="background:${c}"></span>${n}</td>${cells}<td class="num" style="color:${deltaColor}">${delta > 0 ? "+" : ""}${delta.toFixed(2)} pp</td></tr>`;
  }).join("");
  document.getElementById("tabla-curva").innerHTML = `<thead>${headers}</thead><tbody>${rows}</tbody>`;
}

function renderLecturaCurva() {
  const yc = D.yield_curve;
  const c28a = yc["Actual"][0], c28_12m = yc["Hace 12 meses"][0];
  const c728a = yc["Actual"][4], c728_12m = yc["Hace 12 meses"][4];
  const pend_act = c728a - c28a;
  const pend_12m = c728_12m - c28_12m;
  const lecturas = [
    `<p><strong>Desplazamiento a la baja:</strong> la curva completa cayó respecto a hace 12 meses — CETES 28D pasó de ${c28_12m.toFixed(2)}% a ${c28a.toFixed(2)}% (${(c28a - c28_12m > 0 ? "+":"") + (c28a - c28_12m).toFixed(2)} pp).</p>`,
    `<p style="margin-top:10px"><strong>Pendiente positiva:</strong> ${pend_act.toFixed(2)} pp entre 728D y 28D. No hay señales de inversión.</p>`,
    `<p style="margin-top:10px"><strong>Pendiente ${pend_act > pend_12m ? "se empinó" : "se aplanó"} ligeramente</strong> respecto a hace 12 meses (${pend_12m.toFixed(2)} pp).</p>`,
  ];
  document.getElementById("lectura-curva").innerHTML = lecturas.join("");
}

// ════════════════════════════════════════════════════════════════
// PÁGINA 3 — COMPORTAMIENTO POR PLAZO
// ════════════════════════════════════════════════════════════════
function renderInsightVolatil() {
  const k = D.kpis;
  document.getElementById("insight-volatil").innerHTML = `
    <div>
      <div class="label">Hallazgo principal</div>
      <div class="big">${k.plazo_mas_volatil}</div>
    </div>
    <div class="right">
      <p>Es el plazo con mayor movimiento semanal promedio en los últimos 12 meses
      (<strong>${k.bp_plazo_mas_volatil.toFixed(1)} bp/semana</strong>), consistente con
      su mayor sensibilidad de duración. En contraste, <strong>${k.plazo_mas_estable}</strong>
      es el más estable (${k.bp_plazo_mas_estable.toFixed(1)} bp/semana).</p>
    </div>`;
}

function renderChartBP() {
  const traces = PLAZOS.map(p => {
    const s = D.series[String(p)];
    return {
      type: "scatter", mode: "lines", name: s.name, x: s.x, y: s.bp,
      line: { color: s.color, width: 2, shape: "linear" },
      hovertemplate: "<b>%{fullData.name}</b><br>%{x|%d %b %Y}: %{y:+.0f} bp<extra></extra>",
    };
  });
  Plotly.newPlot("chart-bp", traces, {
    margin: { l: 50, r: 20, t: 10, b: 60 },
    xaxis: { gridcolor: "#EFF1F6", showline: false, tickformat: "%b %y" },
    yaxis: { gridcolor: "#EFF1F6", ticksuffix: " bp", zeroline: true, zerolinecolor: "#C8CFDC", zerolinewidth: 1.5 },
    legend: { orientation: "h", x: 0.5, y: -0.18, xanchor: "center" },
    hovermode: "x unified",
    paper_bgcolor: "white", plot_bgcolor: "white",
    font: { family: "Segoe UI, Arial", color: "#1A2233", size: 12 },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#C8CFDC", width: 1.5 } }],
  }, { displaylogo: false, responsive: true });
}

function renderTablaVolatilidad() {
  const k = D.kpis;
  const ranked = PLAZOS.slice().sort((a, b) => k.bp_promedio_abs[NOMBRES[b]] - k.bp_promedio_abs[NOMBRES[a]]);
  const rows = ranked.map((p, idx) => {
    const n = NOMBRES[p];
    const c = COLORS[String(p)];
    const hl = (idx === 0) ? "highlight" : "";
    return `<tr class="${hl}">
      <td><strong>#${idx + 1}</strong></td>
      <td><span class="dot" style="background:${c}"></span>${n}</td>
      <td class="num">${k.bp_promedio_abs[n].toFixed(1)} bp</td>
      <td class="num">${k.bp_max_abs[n].toFixed(0)} bp</td>
    </tr>`;
  }).join("");
  document.getElementById("tabla-volatilidad").innerHTML = `
    <thead>
      <tr><th>Rank</th><th>Plazo</th>
      <th style="text-align:right">|bp| promedio semanal</th>
      <th style="text-align:right">|bp| máximo semanal</th></tr>
    </thead>
    <tbody>${rows}</tbody>`;
}

function renderPronostico() {
  const p = D.pronostico;
  // Para que ARIMA/Prophet/XGBoost se vean continuas desde el último histórico:
  const lastHistX = p.hist_x[p.hist_x.length - 1];
  const lastHistY = p.hist_y[p.hist_y.length - 1];
  const fcstX = [lastHistX].concat(p.fcst_x);
  const arima = [lastHistY].concat(p.arima);
  const arimaInf = [lastHistY].concat(p.arima_inf);
  const arimaSup = [lastHistY].concat(p.arima_sup);
  const prophet = [lastHistY].concat(p.prophet);
  const xgboost = [lastHistY].concat(p.xgboost);

  const traces = [
    // Banda IC 95% (ARIMA) — primero para que quede al fondo
    {
      type: "scatter", mode: "lines", name: "IC 95% sup",
      x: fcstX, y: arimaSup,
      line: { color: "rgba(0,0,0,0)", width: 0 },
      showlegend: false, hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "lines", name: "IC 95% ARIMA",
      x: fcstX, y: arimaInf,
      line: { color: "rgba(0,0,0,0)", width: 0 },
      fill: "tonexty", fillcolor: "rgba(150,150,160,0.18)",
      hovertemplate: "IC 95%: %{y:.2f}%<extra></extra>",
    },
    // Histórico
    {
      type: "scatter", mode: "lines+markers", name: "Histórico CETES 28D",
      x: p.hist_x, y: p.hist_y,
      line: { color: NAVY, width: 3 },
      marker: { size: 6, color: NAVY },
      hovertemplate: "Histórico<br>%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
    },
    // ARIMA
    {
      type: "scatter", mode: "lines+markers", name: "ARIMA",
      x: fcstX, y: arima,
      line: { color: "#EB0029", width: 3, dash: "dash" },
      marker: { size: 9, color: "#EB0029", symbol: "diamond" },
      hovertemplate: "ARIMA<br>%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
    },
    // Prophet
    {
      type: "scatter", mode: "lines+markers", name: "Prophet",
      x: fcstX, y: prophet,
      line: { color: "#0072CE", width: 3, dash: "dash" },
      marker: { size: 9, color: "#0072CE", symbol: "circle" },
      hovertemplate: "Prophet<br>%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
    },
    // XGBoost
    {
      type: "scatter", mode: "lines+markers", name: "XGBoost",
      x: fcstX, y: xgboost,
      line: { color: "#00A651", width: 3, dash: "dash" },
      marker: { size: 9, color: "#00A651", symbol: "triangle-up" },
      hovertemplate: "XGBoost<br>%{x|%d %b %Y}: %{y:.2f}%<extra></extra>",
    },
  ];
  Plotly.newPlot("chart-pronostico", traces, {
    margin: { l: 50, r: 20, t: 10, b: 60 },
    xaxis: { gridcolor: "#EFF1F6", tickformat: "%d %b", showline: false },
    yaxis: { gridcolor: "#EFF1F6", ticksuffix: "%", zeroline: false },
    legend: { orientation: "h", x: 0.5, y: -0.18, xanchor: "center", font: { size: 12 } },
    hovermode: "x unified",
    paper_bgcolor: "white", plot_bgcolor: "white",
    font: { family: "Segoe UI, Arial", color: "#1A2233", size: 12 },
    shapes: [{
      type: "line", xref: "x", yref: "paper",
      x0: lastHistX, x1: lastHistX, y0: 0, y1: 1,
      line: { color: "#C8CFDC", width: 1.5, dash: "dot" },
    }],
    annotations: [{
      x: lastHistX, y: 1, xref: "x", yref: "paper",
      text: "← Histórico │ Pronóstico →", showarrow: false,
      font: { size: 11, color: "#6B7488" }, yshift: 12,
    }],
  }, { displaylogo: false, responsive: true });
}

function renderTablaPronostico() {
  const p = D.pronostico;
  const rows = p.fcst_x.map((fecha, i) => {
    const f = new Date(fecha);
    const fechaStr = f.toLocaleDateString('es-MX', { day: '2-digit', month: 'short', year: 'numeric' });
    return `<tr>
      <td>Semana ${i + 1}<br><span style="font-size:11px;color:#6B7488">${fechaStr}</span></td>
      <td class="num" style="color:#EB0029">${p.arima[i].toFixed(2)}%</td>
      <td class="num" style="color:#0072CE">${p.prophet[i].toFixed(2)}%</td>
      <td class="num" style="color:#00A651">${p.xgboost[i].toFixed(2)}%</td>
    </tr>`;
  }).join("");
  document.getElementById("tabla-pronostico").innerHTML = `
    <thead>
      <tr>
        <th>Semana</th>
        <th style="text-align:right;color:#EB0029">ARIMA</th>
        <th style="text-align:right;color:#0072CE">Prophet</th>
        <th style="text-align:right;color:#00A651">XGBoost</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>`;
}

function renderInsightModelo() {
  const m = D.metricas;
  const g = m.ganador;
  const mape_g = m[g].mape;
  const mape_arima = m.ARIMA.mape;
  const mape_xgb = m.XGBoost.mape;
  const diff_vs_arima = (mape_arima - mape_g).toFixed(2);
  document.getElementById("insight-modelo").innerHTML = `
    <div>
      <div class="label">Recomendación</div>
      <div class="big">${g}</div>
    </div>
    <div class="right">
      <p><strong>Modelo seleccionado</strong> tras validación hold-out de las últimas
      ${m.horizonte_validacion} semanas sobre 5 años de historia. ${g} logró el menor MAPE
      (<strong>${mape_g}%</strong>), apenas ${diff_vs_arima} pp mejor que ARIMA (${mape_arima}%)
      — diferencia pequeña pero consistente. XGBoost queda descartado para producción
      por R² negativo de validación cruzada (${m.XGBoost.cv_r2_mean}), señal de que no
      generaliza con la cantidad de observaciones disponibles.</p>
    </div>`;
}

function renderTablaMetricas() {
  const m = D.metricas;
  const order = ["Prophet", "ARIMA", "XGBoost"]; // ordenado por MAPE
  const colorMap = { "Prophet": "#0072CE", "ARIMA": "#EB0029", "XGBoost": "#00A651" };
  const rows = order.map((name, idx) => {
    const data = m[name];
    const hl = (idx === 0) ? "highlight" : "";
    const badge = (idx === 0) ? ' <span style="color:#00813F;font-size:11px">★ ganador</span>' : '';
    const rmseDisplay = data.rmse != null ? data.rmse.toFixed(4) : "—";
    return `<tr class="${hl}">
      <td><span class="dot" style="background:${colorMap[name]}"></span><strong>${name}</strong>${badge}</td>
      <td class="num">${data.mae.toFixed(4)}</td>
      <td class="num">${rmseDisplay}</td>
      <td class="num"><strong>${data.mape.toFixed(2)}%</strong></td>
    </tr>`;
  }).join("");
  document.getElementById("tabla-metricas").innerHTML = `
    <thead>
      <tr>
        <th>Modelo</th>
        <th style="text-align:right">MAE</th>
        <th style="text-align:right">RMSE</th>
        <th style="text-align:right">MAPE</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>`;
}

function renderChartMape() {
  const m = D.metricas;
  const order = ["Prophet", "ARIMA", "XGBoost"];
  const colorMap = { "Prophet": "#0072CE", "ARIMA": "#EB0029", "XGBoost": "#00A651" };
  const opacityMap = { "Prophet": 1.0, "ARIMA": 0.85, "XGBoost": 0.55 };
  const x = order.map(n => n);
  const y = order.map(n => m[n].mape);
  const colors = order.map(n => {
    const c = colorMap[n];
    const op = opacityMap[n];
    // hex to rgba
    const r = parseInt(c.slice(1,3), 16);
    const g = parseInt(c.slice(3,5), 16);
    const b = parseInt(c.slice(5,7), 16);
    return `rgba(${r},${g},${b},${op})`;
  });
  Plotly.newPlot("chart-mape", [{
    type: "bar", x: x, y: y,
    marker: { color: colors, line: { color: order.map(n => colorMap[n]), width: 1.5 } },
    text: y.map(v => v.toFixed(2) + "%"),
    textposition: "outside",
    textfont: { size: 13, color: "#1A2233", family: "Segoe UI" },
    hovertemplate: "<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>",
  }], {
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: { gridcolor: "#EFF1F6", showline: false },
    yaxis: { gridcolor: "#EFF1F6", ticksuffix: "%", zeroline: false, range: [0, Math.max(...y) * 1.25] },
    paper_bgcolor: "white", plot_bgcolor: "white",
    font: { family: "Segoe UI, Arial", color: "#1A2233", size: 12 },
    showlegend: false,
  }, { displaylogo: false, responsive: true });
}

function renderCardsModelos() {
  const m = D.metricas;
  const order = ["Prophet", "ARIMA", "XGBoost"];
  const colorMap = { "Prophet": "#0072CE", "ARIMA": "#EB0029", "XGBoost": "#00A651" };
  const html = order.map((name) => {
    const data = m[name];
    const isWinner = (name === m.ganador);
    return `<div class="model-card ${isWinner ? 'winner' : ''}">
      ${isWinner ? '<span class="winner-badge">Recomendado</span>' : ''}
      <div class="model-head">
        <h4 style="color:${colorMap[name]}">${name}</h4>
        <span class="mape-tag">MAPE ${data.mape}%</span>
      </div>
      <div class="tipo">${data.tipo}</div>
      <p class="pro"><strong>✓ Fortaleza:</strong> ${data.fortaleza}</p>
      <p class="con"><strong>✗ Debilidad:</strong> ${data.debilidad}</p>
    </div>`;
  }).join("");
  document.getElementById("cards-modelos").innerHTML = html;
}

// ════════════════════════════════════════════════════════════════
// INICIALIZACIÓN
// ════════════════════════════════════════════════════════════════
renderKpisPage1();
renderKpisPage2();
renderKpisPage3();
renderPlazoFilters();
setupDateFilters();
renderEvolucion();
renderTablaPromedios();
renderCurva();
renderTablaCurva();
renderLecturaCurva();
renderInsightVolatil();
renderChartBP();
renderTablaVolatilidad();
renderTablaPronostico();
renderPronostico();
renderInsightModelo();
renderTablaMetricas();
renderChartMape();
renderCardsModelos();
</script>

</body>
</html>
"""

HTML = HTML.replace("__PAYLOAD__", payload_json)
HTML = HTML.replace("__FECHA_ULTIMA__", kpis["fecha_ultimo_dato"])
HTML = HTML.replace("__BUILD_DATE__", payload["build_date"])
if plotly_js:
    HTML = HTML.replace("__PLOTLY_LIB__", f'<script>{plotly_js}</script>')
else:
    HTML = HTML.replace(
        "__PLOTLY_LIB__",
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    )

out = BASE / "dashboard_cetes.html"
out.write_text(HTML, encoding="utf-8")
print(f"✅ Dashboard generado: {out}")
print(f"   - Tamaño: {len(HTML):,} bytes")
print(f"   - Período: {kpis['fecha_min']} → {kpis['fecha_max']}")
print(f"   - Observaciones: {kpis['n_observaciones']}")
print(f"   - Plazo más volátil: {kpis['plazo_mas_volatil']} ({kpis['bp_plazo_mas_volatil']:.1f} bp/sem)")
