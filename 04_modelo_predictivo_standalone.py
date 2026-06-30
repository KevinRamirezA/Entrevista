from dotenv import load_dotenv
import os
load_dotenv()

#!/usr/bin/env python3
"""
================================================================================
ETAPA 4: MODELO PREDICTIVO CETES 28D — Standalone
AFORE XXI Banorte - Prueba Técnica Analista de Data Science
================================================================================

Este script es completamente autónomo:
  1. Extrae CETES 28D directamente de la API de Banxico (últimos 5 años)
  2. Limpia y prepara la serie temporal
  3. Entrena y compara 3 modelos: ARIMA, Prophet, XGBoost
  4. Exporta todas las predicciones en un único CSV comparativo

No requiere base de datos ni scripts previos.

Salida: predicciones_cetes_28d_comparativa.csv
"""

import requests
import pandas as pd
import numpy as np
import logging
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('pipeline_cetes.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIE.Predictor")

# ── CONFIGURACIÓN ───────────────────────────────────────────────────────────
TOKEN = os.getenv("BANXICO_TOKEN")
SERIE_28D = "SF43936"          # ID de CETES 28 días en Banxico
OUTPUT_CSV = "predicciones_cetes_28d_comparativa.csv"

# ── IMPORTS OPCIONALES ──────────────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️  pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  pip install prophet")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  pip install xgboost")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DIRECTA DE LA API
# ═══════════════════════════════════════════════════════════════════════════════
class SIEApiClient:
    """Cliente mínimo para extraer CETES 28D de la API de Banxico."""

    BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1"
    MAX_RETRIES = 4
    BASE_DELAY = 1.0

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"Bmx-Token": token, "Accept": "application/json"})

    def _backoff(self, attempt: int) -> float:
        return 2 ** attempt + random.uniform(0, 1)

    def _make_request(self, endpoint: str) -> dict:
        url = f"{self.BASE_URL}/{endpoint}"
        for attempt in range(self.MAX_RETRIES):
            try:
                time.sleep(self.BASE_DELAY)
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = self._backoff(attempt + 1)
                    logger.warning(f"Rate limit. Esperando {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error intento {attempt+1}: {e}")
                time.sleep(self._backoff(attempt))
        logger.error(f"Máximo reintentos: {url}")
        return None

    def obtener_cetes_28d(self, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
        """Extrae CETES 28D y retorna DataFrame limpio."""
        endpoint = f"series/{SERIE_28D}/datos/{fecha_inicio}/{fecha_fin}"
        data = self._make_request(endpoint)
        if not data:
            raise ValueError("No se pudieron obtener datos de la API")

        registros = []
        for serie in data.get("bmx", {}).get("series", []):
            for dato in serie.get("datos", []):
                raw_val = dato.get("dato", "N/E")
                registros.append({
                    "fecha": dato.get("fecha"),      # DD/MM/YYYY
                    "tasa_rendimiento": None if raw_val == "N/E" else float(raw_val),
                    "es_ne": raw_val == "N/E",
                })

        df = pd.DataFrame(registros)
        # Parsear fecha
        df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
        # Ordenar y establecer frecuencia semanal (jueves)
        df = df.sort_values("fecha").set_index("fecha").asfreq("W-THU")
        # Forward fill para N/E
        df["tasa_rendimiento"] = df["tasa_rendimiento"].ffill()
        logger.info(f"Datos extraídos: {len(df)} semanas ({df.index.min().date()} → {df.index.max().date()})")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS PREDICTIVOS
# ═══════════════════════════════════════════════════════════════════════════════
class CETESPredictor:
    def __init__(self, df: pd.DataFrame):
        self.df_28 = df
        self.model_arima = None
        self.model_prophet = None
        self.model_xgb = None
        self.features = None
        self.mejor_modelo = None
        self.mejor_mape = float('inf')

    def _crear_features_xgb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features para XGBoost: lags, medias móviles, calendario."""
        df_m = df.reset_index().copy()
        df_m.columns = ["fecha", "tasa_rendimiento", "es_ne"]

        for lag in [1, 2, 3, 4]:
            df_m[f"lag_{lag}"] = df_m["tasa_rendimiento"].shift(lag)

        df_m["ma_3"] = df_m["tasa_rendimiento"].rolling(3, min_periods=1).mean().shift(1)
        df_m["ma_8"] = df_m["tasa_rendimiento"].rolling(8, min_periods=1).mean().shift(1)
        df_m["ma_12"] = df_m["tasa_rendimiento"].rolling(12, min_periods=1).mean().shift(1)
        df_m["std_4"] = df_m["tasa_rendimiento"].rolling(4, min_periods=2).std().shift(1)
        df_m["diff_1"] = df_m["tasa_rendimiento"].diff(1).shift(1)
        df_m["diff_4"] = df_m["tasa_rendimiento"].diff(4).shift(1)
        df_m["mes"] = df_m["fecha"].dt.month
        df_m["trimestre"] = df_m["fecha"].dt.quarter
        df_m["anio"] = df_m["fecha"].dt.year

        return df_m.dropna(subset=["tasa_rendimiento", "lag_1", "lag_2"])

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    def _entrenar_arima(self) -> dict:
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels no instalado"}

        serie = self.df_28["tasa_rendimiento"].dropna()
        if len(serie) < 20:
            return {"error": f"Solo {len(serie)} observaciones"}

        n_test = min(12, len(serie) // 4)
        train, test = serie.iloc[:-n_test], serie.iloc[-n_test:]

        try:
            model = ARIMA(train, order=(1, 1, 1))
            self.model_arima = model.fit()
        except Exception as e:
            return {"error": str(e)}

        pred_test = self.model_arima.forecast(steps=len(test))
        mae = mean_absolute_error(test, pred_test)
        rmse = np.sqrt(mean_squared_error(test, pred_test))
        mape = np.mean(np.abs((test - pred_test) / test)) * 100

        self.model_arima = ARIMA(serie, order=(1, 1, 1)).fit()

        if mape < self.mejor_mape:
            self.mejor_mape = mape
            self.mejor_modelo = "ARIMA"

        return {"modelo": "ARIMA", "mae": round(mae, 4), "rmse": round(rmse, 4),
                "mape": round(mape, 2), "aic": round(self.model_arima.aic, 2)}

    def _predecir_arima(self, horizonte: int = 4) -> pd.DataFrame:
        pred = self.model_arima.get_forecast(steps=horizonte)
        pred_mean = pred.predicted_mean
        pred_ci = pred.conf_int()
        fechas = pd.date_range(
            start=self.df_28.index[-1] + timedelta(weeks=1),
            periods=horizonte, freq="W-THU"
        )
        return pd.DataFrame({
            "fecha": fechas,
            "prediccion_arima": np.round(pred_mean.values, 4),
            "arima_inf_95": np.round(pred_ci.iloc[:, 0].values, 4),
            "arima_sup_95": np.round(pred_ci.iloc[:, 1].values, 4),
        })

    # ── PROPHET ───────────────────────────────────────────────────────────────
    def _entrenar_prophet(self) -> dict:
        if not PROPHET_AVAILABLE:
            return {"error": "Prophet no instalado"}

        df_p = self.df_28.reset_index().rename(columns={"fecha": "ds", "tasa_rendimiento": "y"})
        n_test = min(12, len(df_p) // 4)
        train_df = df_p.iloc[:-n_test]
        test_df = df_p.iloc[-n_test:]

        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                            daily_seasonality=False, changepoint_prior_scale=0.05,
                            interval_width=0.95)
            model.fit(train_df)
            self.model_prophet = model

            future = model.make_future_dataframe(periods=n_test, freq="W-THU")
            forecast = model.predict(future)
            pred_test = forecast.iloc[-n_test:]["yhat"].values

            mae = mean_absolute_error(test_df["y"], pred_test)
            rmse = np.sqrt(mean_squared_error(test_df["y"], pred_test))
            mape = np.mean(np.abs((test_df["y"] - pred_test) / test_df["y"])) * 100

            model_final = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                                  daily_seasonality=False, changepoint_prior_scale=0.05,
                                  interval_width=0.95)
            model_final.fit(df_p)
            self.model_prophet = model_final

            if mape < self.mejor_mape:
                self.mejor_mape = mape
                self.mejor_modelo = "Prophet"

            return {"modelo": "Prophet", "mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 2)}
        except Exception as e:
            return {"error": str(e)}

    def _predecir_prophet(self, horizonte: int = 4) -> pd.DataFrame:
        future = self.model_prophet.make_future_dataframe(periods=horizonte, freq="W-THU")
        forecast = self.model_prophet.predict(future)
        pred = forecast.iloc[-horizonte:]
        return pd.DataFrame({
            "fecha": pred["ds"].values,
            "prediccion_prophet": np.round(pred["yhat"].values, 4),
            "prophet_inf_95": np.round(pred["yhat_lower"].values, 4),
            "prophet_sup_95": np.round(pred["yhat_upper"].values, 4),
        })

    # ── XGBOOST ───────────────────────────────────────────────────────────────
    def _entrenar_xgboost(self) -> dict:
        if not XGBOOST_AVAILABLE:
            return {"error": "XGBoost no instalado"}

        df_m = self._crear_features_xgb(self.df_28)
        if len(df_m) < 15:
            return {"error": f"Solo {len(df_m)} muestras"}

        self.features = [c for c in df_m.columns if c not in ["fecha", "tasa_rendimiento", "es_ne"]]
        X, y = df_m[self.features], df_m["tasa_rendimiento"]
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())

        n_splits = min(5, len(X) // 10)
        if n_splits < 2:
            n_splits = 2

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores_r2, scores_mae = [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            scores_r2.append(r2_score(y_te, pred))
            scores_mae.append(mean_absolute_error(y_te, pred))

        self.model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                          subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.model_xgb.fit(X, y)

        mape_cv = np.mean([mae / y.mean() * 100 for mae in scores_mae])
        if mape_cv < self.mejor_mape:
            self.mejor_mape = mape_cv
            self.mejor_modelo = "XGBoost"

        return {"modelo": "XGBoost", "cv_r2_mean": round(np.mean(scores_r2), 4),
                "cv_mae_mean": round(np.mean(scores_mae), 4), "mape": round(mape_cv, 2)}

    def _predecir_xgboost(self, horizonte: int = 4) -> pd.DataFrame:
        df_m = self._crear_features_xgb(self.df_28)
        last_row = df_m.iloc[-1:][self.features].copy()
        preds, dates = [], []
        historial = list(self.df_28["tasa_rendimiento"].values)

        for i in range(horizonte):
            pred = self.model_xgb.predict(last_row)[0]
            preds.append(round(pred, 4))
            next_date = df_m["fecha"].iloc[-1] + timedelta(weeks=i+1)
            dates.append(next_date)
            historial.append(pred)

            new_row = last_row.copy()
            for lag in range(4, 1, -1):
                new_row[f"lag_{lag}"] = new_row[f"lag_{lag-1}"]
            new_row["lag_1"] = pred
            new_row["ma_3"] = np.mean(historial[-3:])
            new_row["ma_8"] = np.mean(historial[-8:]) if len(historial) >= 8 else np.mean(historial)
            new_row["ma_12"] = np.mean(historial[-12:]) if len(historial) >= 12 else np.mean(historial)
            new_row["std_4"] = np.std(historial[-4:]) if len(historial) >= 4 else 0
            new_row["diff_1"] = historial[-1] - historial[-2]
            new_row["diff_4"] = historial[-1] - historial[-5] if len(historial) >= 5 else 0
            new_row["mes"] = next_date.month
            new_row["trimestre"] = (next_date.month - 1) // 3 + 1
            new_row["anio"] = next_date.year
            last_row = new_row

        return pd.DataFrame({"fecha": dates, "prediccion_xgboost": preds})

    # ── ORQUESTADOR ──────────────────────────────────────────────────────────
    def entrenar(self) -> dict:
        resultados = {}
        if STATSMODELS_AVAILABLE:
            resultados["arima"] = self._entrenar_arima()
        if PROPHET_AVAILABLE:
            resultados["prophet"] = self._entrenar_prophet()
        if XGBOOST_AVAILABLE:
            resultados["xgboost"] = self._entrenar_xgboost()
        return resultados

    def predecir_todos(self, horizonte: int = 4) -> pd.DataFrame:
        """Genera un único CSV con predicciones de todos los modelos."""
        dfs = []

        if self.model_arima is not None:
            dfs.append(self._predecir_arima(horizonte))
        if self.model_prophet is not None:
            dfs.append(self._predecir_prophet(horizonte))
        if self.model_xgb is not None:
            dfs.append(self._predecir_xgboost(horizonte))

        if not dfs:
            raise ValueError("Ningún modelo entrenado")

        resultado = dfs[0]
        for df in dfs[1:]:
            resultado = resultado.merge(df, on="fecha", how="outer")

        return resultado.sort_values("fecha").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("MODELO PREDICTIVO CETES 28D — Standalone")
    print("="*60)

    # 1. EXTRAER DATOS DE LA API
    print("\n📡 Extrayendo CETES 28D de Banxico...")
    client = SIEApiClient(TOKEN)

    # Últimos 5 años de datos (suficiente para modelos robustos)
    fecha_fin = datetime.now().strftime("%Y-%m-%d")
    fecha_inicio = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")

    df = client.obtener_cetes_28d(fecha_inicio, fecha_fin)
    print(f"Serie cargada: {len(df)} semanas")

    # 2. ENTRENAR MODELOS
    print("\n Entrenando modelos...")
    predictor = CETESPredictor(df)
    metrics = predictor.entrenar()

    print("\n" + "="*60)
    print("MÉTRICAS DE MODELOS")
    print("="*60)

    for nombre, m in metrics.items():
        if "error" not in m:
            print(f"\n {m['modelo']}:")
            for k, v in m.items():
                if k != "modelo":
                    print(f"   {k}: {v}")
        else:
            print(f"\n {nombre.upper()}: {m['error']}")

    print(f"\n MEJOR MODELO: {predictor.mejor_modelo} (MAPE: {predictor.mejor_mape:.2f}%)")

    # 3. GENERAR CSV COMPARATIVO
    print("\n" + "="*60)
    print("PREDICCIONES COMPARATIVAS (4 semanas)")
    print("="*60)
    forecast = predictor.predecir_todos(horizonte=4)
    print(forecast.to_string(index=False))

    forecast.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n Guardado: {OUTPUT_CSV}")

    print("\n" + "="*60)
    print("COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()
