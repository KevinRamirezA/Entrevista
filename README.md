# Pipeline de Rendimientos CETES — AFORE XXI Banorte

## Prueba Técnica: Analista de Data Science

**Subdirección de Trading y Análisis Cuantitativo** | Junio 2026

\---

## Resumen Ejecutivo

Este proyecto implementa un **pipeline ETL completo** para la extracción, procesamiento, análisis y predicción de las tasas de rendimiento de **Certificados de la Tesorería (CETES)** a través del **Sistema de Información Económica (SIE)** del Banco de México.

### Alcance

* **Instrumentos:** CETES a 28, 91, 182, 364 y 728 días
* **Fuentes:** API REST SIE Banxico (series SF43936, SF43939, SF43942, SF43945, SF349785)
* **Período:** Últimos 12 meses de datos históricos + 4 semanas de predicción
* **Modelos:** ARIMA, Prophet (Meta) y XGBoost con validación cruzada temporal
* **Visualización:** Dashboard interactivo HTML con Plotly.js

### KPIs del Pipeline

|Métrica|Valor|
|-|-|
|Registros extraídos|208|
|Cobertura temporal|52 semanas|
|Datos faltantes (N/E)|0.0%|
|Outliers detectados|1|
|Modelos entrenados|3|
|Horizonte predictivo|4 semanas|

\---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FUENTE: API SIE BANXICO                            │
│           https://www.banxico.org.mx/SieAPIRest/service/v1                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 1: EXTRACCIÓN (01\\\\\\\_extraccion\\\\\\\_api\\\\\\\_v2.py)                              │
│  • Autenticación Bmx-Token                                                  │
│  • Backoff exponencial con jitter (rate limiting)                           │
│  • Batching (máx. 20 series/request)                                        │
│  • Validación de token previa                                               │
│  • Checkpoint incremental                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 2: PERSISTENCIA (02\\\\\\\_base\\\\\\\_datos\\\\\\\_v2.py)                                │
│  • SQLite con índices optimizados (plazo, fecha, serie)                       │
│  • UPSERT con manejo de duplicados                                          │
│  • Tabla de auditoría pipeline                                              │
│  • Compatible PostgreSQL/MySQL                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 3: DATA MINING (03\\\\\\\_data\\\\\\\_mining\\\\\\\_v2.py)                                │
│  • Imputación ffill/bfill por plazo                                         │
│  • Detección de outliers (IQR × 1.5)                                        │
│  • Features: variación %, cambio bp, media móvil, volatilidad, z-score      │
│  • Spreads vs CETES 28D (benchmark)                                         │
│  • Pendiente de curva (728D - 28D)                                          │
│  • Exportación: formato largo + wide (Power BI)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 4: MODELADO PREDICTIVO (04\\\\\\\_modelo\\\\\\\_predictivo\\\\\\\_standalone.py)          │
│  • Extracción directa 5 años de historia                                    │
│  • ARIMA(1,1,1) con intervalos de confianza 95%                             │
│  • Prophet: estacionalidad anual, changepoint\\\\\\\_prior\\\\\\\_scale=0.05                │
│  • XGBoost: lags, medias móviles, features temporales                     │
│  • TimeSeriesSplit para validación cruzada                                  │
│  • Selección automática del mejor modelo (MAPE)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ENTREGABLES                                                                │
│  • dashboard\\\\\\\_cetes\\\\\\\_interactivo.html  → Dashboard visual interactivo         │
│  • predicciones\\\\\\\_cetes\\\\\\\_28d\\\\\\\_comparativa.csv → Forecast 3 modelos              │
│  • cetes\\\\\\\_clean.csv / cetes\\\\\\\_wide.csv → Datos procesados                      │
│  • pipeline\\\\\\\_cetes.log → Auditoría completa del pipeline                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

\---

## Estructura del Proyecto

```
.
├── .env                                    # Variables de entorno (token Banxico)
├── README.md                               # Este documento
├── requirements.txt                        # Dependencias Python
│
├── 01\\\\\\\_extraccion\\\\\\\_api.py                 # Cliente API SIE con retry logic
├── 02\\\\\\\_base\\\\\\\_datos.py                     # Capa de persistencia SQLite
├── 03\\\\\\\_data\\\\\\\_mining.py                    # Pipeline de limpieza y features
├── 04\\\\\\\_modelo\\\\\\\_predictivo\\\\\\\_standalone.py      # Modelos ARIMA / Prophet / XGBoost
│
├── datos\\\\\\\_cetes\\\\\\\_raw.csv                     # Datos crudos extraídos (208 registros)
├── cetes\\\\\\\_database.db                       # Base de datos SQLite
├── cetes\\\\\\\_clean.csv                         # Datos limpios con features
├── cetes\\\\\\\_wide.csv                          # Formato pivot para BI
├── predicciones\\\\\\\_cetes\\\\\\\_28d\\\\\\\_comparativa.csv  # Forecast comparativo 4 semanas
│
├── pipeline\\\_cetes.log                      # Log de ejecución completo
└── dashboard\\\_cetes.html        # Dashboard interactivo Plotly
```

\---

## Requisitos Previos

### Software

* Python 3.10+
* pip o conda
* Navegador web moderno (para dashboard)

### Credenciales

* Token de acceso a la API SIE de Banxico

  * Solicitud: [https://www.banxico.org.mx/SieAPIRest/service/v1/token](https://www.banxico.org.mx/SieAPIRest/service/v1/token)

### Dependencias Principales

|Paquete|Versión|Uso|
|-|-|-|
|pandas|≥2.0.0|Manipulación de datos|
|numpy|≥1.24.0|Cálculos numéricos|
|requests|≥2.31.0|Consumo API REST|
|statsmodels|≥0.14.0|Modelo ARIMA|
|prophet|≥1.1.5|Modelo Prophet|
|xgboost|≥2.0.0|Modelo XGBoost|
|scikit-learn|≥1.3.0|Métricas y validación|
|python-dotenv|≥1.0.0|Gestión de variables de entorno|

\---

## Instalación y Configuración

### 1\. Clonar o descargar el proyecto

```bash
cd pipeline-cetes-banorte
```

### 2\. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\\\\\\\\Scripts\\\\\\\\activate       # Windows
```

### 3\. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4\. Configurar credenciales

Crear archivo `.env` en la raíz del proyecto:

```bash
echo "BANXICO\\\\\\\_TOKEN=tu\\\\\\\_token\\\\\\\_aqui" > .env
```

> ⚠️ \\\\\\\*\\\\\\\*Seguridad:\\\\\\\*\\\\\\\* El archivo `.env` está en `.gitignore` por defecto. Nunca commitear tokens.

\---

## Ejecución del Pipeline

### Opción A: Ejecución Completa (Orquestada)

```bash
# Etapa 1: Extracción
python 01\\\\\\\_extraccion\\\\\\\_api\\\\\\\_v2.py

# Etapa 2: Persistencia
python 02\\\\\\\_base\\\\\\\_datos\\\\\\\_v2.py

# Etapa 3: Data Mining
python 03\\\\\\\_data\\\\\\\_mining\\\\\\\_v2.py

# Etapa 4: Modelado Predictivo
python 04\\\\\\\_modelo\\\\\\\_predictivo\\\\\\\_standalone.py

\# Etapa 5: dashboard\_cetes
python 05 build\_dashboard.py
```



### Opción B: Ejecución Standalone (Modelo Predictivo)

El script `04\\\\\\\_modelo\\\\\\\_predictivo\\\\\\\_standalone.py` es **autónomo** y puede ejecutarse sin las etapas previas:

```bash
python 04\\\\\\\_modelo\\\\\\\_predictivo\\\\\\\_standalone.py
```

Este script extrae directamente 5 años de historia, entrena los 3 modelos y genera el CSV comparativo.

### Verificación

Al finalizar, revisar:

* `pipeline\\\\\\\_cetes.log` — Log completo de ejecución
* `predicciones\\\\\\\_cetes\\\\\\\_28d\\\\\\\_comparativa.csv` — Predicciones
* `dashboard\\\\\\\_cetes\\\\\\\_interactivo.html` — Dashboard visual

\---

## Descripción de Etapas

### Etapa 1: Extracción API (`01\\\\\\\_extraccion\\\\\\\_api\\\\\\\_v2.py`)

**Clase principal:** `SIEApiClient`

|Característica|Implementación|
|-|-|
|Autenticación|Header `Bmx-Token`|
|Rate limiting|Delay base 1.0s + backoff exponencial|
|Retry logic|Máx. 4 intentos con jitter aleatorio|
|Batch processing|Hasta 20 series por request|
|Validación|Token testeado antes de extracción masiva|
|Cobertura|Últimos 12 meses (configurable)|

**Series consultadas:**

|ID Serie|Plazo|Frecuencia|Descripción|
|-|-|-|-|
|SF43936|28 días|Semanal|CETES 28 días|
|SF43939|91 días|Semanal|CETES 91 días|
|SF43942|182 días|Semanal|CETES 182 días|
|SF43945|364 días|Semanal|CETES 364 días|
|SF349785|728 días|Quincenal|CETES 728 días|

**Salida:** `datos\\\\\\\_cetes\\\\\\\_raw.csv`

\---

### Etapa 2: Base de Datos (`02\\\\\\\_base\\\\\\\_datos\\\\\\\_v2.py`)

**Clase principal:** `CETESDatabase`

**Esquema relacional:**

```sql
CREATE TABLE cetes\\\\\\\_rendimientos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    serie\\\\\\\_id TEXT NOT NULL,
    plazo\\\\\\\_dias INTEGER NOT NULL,
    nombre\\\\\\\_serie TEXT NOT NULL,
    fecha DATE NOT NULL,
    tasa\\\\\\\_rendimiento REAL,
    es\\\\\\\_dato\\\\\\\_faltante BOOLEAN DEFAULT 0,
    frecuencia TEXT NOT NULL,
    unidad TEXT NOT NULL,
    fecha\\\\\\\_extraccion TIMESTAMP DEFAULT CURRENT\\\\\\\_TIMESTAMP,
    UNIQUE(serie\\\\\\\_id, fecha)
);

CREATE TABLE auditoria\\\\\\\_pipeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    etapa TEXT NOT NULL,
    registros\\\\\\\_afectados INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT\\\\\\\_TIMESTAMP,
    detalle TEXT
);
```

**Índices:** `idx\\\\\\\_plazo`, `idx\\\\\\\_fecha`, `idx\\\\\\\_plazo\\\\\\\_fecha`, `idx\\\\\\\_serie`

**Salida:** `cetes\\\\\\\_database.db`

\---

### Etapa 3: Data Mining (`03\\\\\\\_data\\\\\\\_mining\\\\\\\_v2.py`)

**Clase principal:** `CETESDataMining`

**Pipeline de transformaciones:**

```
Carga desde SQLite
    ↓
Estandarización (serie\\\\\\\_estandar, tipo\\\\\\\_instrumento, categoria\\\\\\\_plazo)
    ↓
Tratamiento de faltantes (ffill/bfill por plazo, flag dato\\\\\\\_imputado)
    ↓
Cálculo de features:
  • variacion\\\\\\\_pct    → Cambio porcentual semanal
  • cambio\\\\\\\_bp        → Cambio en puntos base
  • media\\\\\\\_movil\\\\\\\_4s   → Media móvil 4 semanas
  • volatilidad\\\\\\\_4s   → Desviación estándar móvil
  • z\\\\\\\_score          → Normalización estadística
  • pendiente\\\\\\\_curva  → Spread 728D - 28D
    ↓
Detección de outliers (IQR × 1.5 por plazo)
    ↓
Cálculo de spreads vs CETES 28D (benchmark)
    ↓
Exportación: largo (cetes\\\\\\\_clean.csv) + wide (cetes\\\\\\\_wide.csv)
```

**Salidas:** `cetes\\\\\\\_clean.csv`, `cetes\\\\\\\_wide.csv`

\---

### Etapa 4: Modelado Predictivo (`04\\\\\\\_modelo\\\\\\\_predictivo\\\\\\\_standalone.py`)

**Clase principal:** `CETESPredictor`

#### Modelos Implementados

|Modelo|Librería|Hiperparámetros|Métricas|
|-|-|-|-|
|**ARIMA**|statsmodels|order=(1,1,1)|MAE, RMSE, MAPE, AIC|
|**Prophet**|prophet|yearly\_seasonality=True, changepoint\_prior\_scale=0.05|MAE, RMSE, MAPE|
|**XGBoost**|xgboost|n\_estimators=100, max\_depth=3, learning\_rate=0.1|CV R², CV MAE, MAPE|

#### Features XGBoost

* Lags: t-1, t-2, t-3, t-4
* Medias móviles: 3, 8, 12 semanas
* Volatilidad: std(4 semanas)
* Diferencias: Δt-1, Δt-4
* Temporal: mes, trimestre, año

#### Estrategia de Validación

* **Train/Test split:** Últimas 12 semanas como test
* **XGBoost:** TimeSeriesSplit con 5 folds (adaptativo)
* **Selección:** Mejor modelo por MAPE mínimo

**Salida:** `predicciones\\\\\\\_cetes\\\\\\\_28d\\\\\\\_comparativa.csv`

\---

## Decisiones Técnicas

### 1\. SQLite vs PostgreSQL/MySQL

**Decisión:** SQLite para prototipo, esquema compatible con PostgreSQL/MySQL.
**Justificación:** Portabilidad, cero configuración, suficiente para <10k registros.

### 2\. Formato Largo vs Ancho

**Decisión:** Almacenamiento en formato largo; exportación dual (largo + ancho).
**Justificación:** Normalización de base de datos + compatibilidad con herramientas BI.

### 3\. Tres Modelos vs Uno

**Decisión:** Ensemble de ARIMA (estadístico), Prophet (estacionalidad) y XGBoost (ML).
**Justificación:** Diversificación de hipótesis; consenso reduce sesgo de un solo modelo.

### 4\. Prophet sobre SARIMA

**Decisión:** Prophet como modelo de estacionalidad.
**Justificación:** Manejo automático de missing values, changepoints y festivos; robustez superior en series financieras irregulares.

### 5\. Dashboard HTML vs Power BI

**Decisión:** Dashboard HTML estático con Plotly.js.
**Justificación:** Portabilidad (no requiere licencia), embebible, control total del diseño.

\---

## Resultados y Entregables

### Dashboard Interactivo

**Archivo:** `dashboard\\\\\\\_cetes\\\\\\\_interactivo.html`

**Pestañas:**

1. **Contexto de Tasas** — Evolución temporal, cambios bp, volatilidad
2. **Yield Curve** — Curva de rendimientos, pendiente, cambios anuales
3. **Predicciones** — Forecast 3 modelos, intervalos de confianza, tabla comparativa

**Tecnología:** Plotly.js (CDN), responsive, sin dependencias de servidor.

## Licencia y Contacto

**Proyecto desarrollado para:** AFORE XXI Banorte — Subdirección de Trading y Análisis Cuantitativo

**Prueba Técnica:** Analista de Data Science | Junio 2026

**Stack tecnológico:** Python, SQLite, Plotly, statsmodels, Prophet, XGBoost

**Datos fuente:** Banco de México — Sistema de Información Económica (SIE)

