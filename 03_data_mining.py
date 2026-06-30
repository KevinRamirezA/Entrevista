from dotenv import load_dotenv
import os
load_dotenv()

#!/usr/bin/env python3
"""
================================================================================
ETAPA 3: DATA MINING Y LIMPIEZA
AFORE XXI Banorte - Prueba Técnica Analista de Data Science
================================================================================
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('pipeline_cetes.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIE.DataMining")

DB_PATH      = "cetes_database.db"
OUTPUT_PATH  = "cetes_clean.csv"
OUTPUT_WIDE  = "cetes_wide.csv"   # formato pivot para Power BI alternativo


NOMBRE_ESTANDAR = {
    "SF43936":  "CETES_28D",
    "SF43939":  "CETES_91D",
    "SF43942":  "CETES_182D",
    "SF43945":  "CETES_364D",
    "SF349785": "CETES_728D",
}


class CETESDataMining:
    """Pipeline de limpieza y enriquecimiento de datos CETES."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.df: pd.DataFrame = pd.DataFrame()

    # ── 1. Carga ──────────────────────────────────────────────────────────────
    def cargar_datos(self) -> "CETESDataMining":
        with sqlite3.connect(self.db_path) as conn:
            self.df = pd.read_sql_query(
                """
                SELECT serie_id, plazo_dias, nombre_serie, fecha,
                       tasa_rendimiento, es_dato_faltante, frecuencia
                FROM cetes_rendimientos
                ORDER BY plazo_dias, fecha
                """,
                conn,
                parse_dates=["fecha"],
            )
        logger.info(f"Datos cargados: {len(self.df)} registros desde {self.db_path}")
        return self

    # ── 2. Estandarizar ───────────────────────────────────────────────────────
    def estandarizar(self) -> "CETESDataMining":
        self.df["serie_estandar"]    = self.df["serie_id"].map(NOMBRE_ESTANDAR)
        self.df["tipo_instrumento"]  = "CETES"
        self.df["categoria_plazo"]   = pd.cut(
            self.df["plazo_dias"],
            bins=[0, 91, 364, 9999],
            labels=["Corto Plazo", "Mediano Plazo", "Largo Plazo"],
        ).astype(str)
        return self

    # ── 3. Tratar faltantes (N/E) ─────────────────────────────────────────────
    def tratar_faltantes(self) -> "CETESDataMining":
        """Forward fill por plazo; bfill para llenar inicio de serie."""
        self.df["dato_imputado"] = self.df["es_dato_faltante"].astype(bool)
        self.df["tasa_rendimiento"] = (
            self.df.groupby("plazo_dias")["tasa_rendimiento"]
            .transform(lambda x: x.ffill().bfill())
        )
        n_imp = self.df["dato_imputado"].sum()
        logger.info(f"Valores N/E imputados (ffill/bfill): {n_imp}")
        return self

    # ── 4. Campos derivados ───────────────────────────────────────────────────
    def calcular_derivados(self) -> "CETESDataMining":
        self.df = self.df.sort_values(["plazo_dias", "fecha"]).reset_index(drop=True)
        g = self.df.groupby("plazo_dias")["tasa_rendimiento"]

        self.df["variacion_pct"]   = g.pct_change() * 100
        self.df["cambio_bp"]       = g.diff() * 100
        self.df["media_movil_4s"]  = g.transform(lambda x: x.rolling(4, min_periods=1).mean())
        self.df["volatilidad_4s"]  = g.transform(lambda x: x.rolling(4, min_periods=2).std())
        self.df["z_score"]         = g.transform(lambda x: (x - x.mean()) / x.std())

        # Pendiente de la curva: spread entre plazo máximo y mínimo disponibles por fecha
        pivot = self.df.pivot_table(index="fecha", columns="plazo_dias",
                                    values="tasa_rendimiento", aggfunc="mean")
        if 728 in pivot.columns and 28 in pivot.columns:
            slope = (pivot[728] - pivot[28]).rename("pendiente_curva")
            self.df = self.df.merge(slope.reset_index(), on="fecha", how="left")
        else:
            self.df["pendiente_curva"] = np.nan

        return self

    # ── 5. Detectar outliers (IQR) ────────────────────────────────────────────
    def detectar_outliers(self) -> "CETESDataMining":
        """Detecta outliers usando IQR × 1.5 por plazo. Usa transform para evitar perder columnas."""
        if self.df.empty:
            self.df["outlier_limite_inf"] = np.nan
            self.df["outlier_limite_sup"] = np.nan
            self.df["es_outlier"] = 0
            logger.warning("DataFrame vacío — no hay outliers que detectar")
            return self


        q1 = self.df.groupby("plazo_dias")["tasa_rendimiento"].transform(lambda x: x.quantile(0.25))
        q3 = self.df.groupby("plazo_dias")["tasa_rendimiento"].transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1

        self.df["outlier_limite_inf"] = q1 - 1.5 * iqr
        self.df["outlier_limite_sup"] = q3 + 1.5 * iqr
        self.df["es_outlier"] = (
            (self.df["tasa_rendimiento"] < self.df["outlier_limite_inf"]) |
            (self.df["tasa_rendimiento"] > self.df["outlier_limite_sup"])
        ).astype(int)

        n_out = self.df["es_outlier"].sum()
        logger.info(f"Outliers detectados (IQR ×1.5): {n_out}")
        return self

    # ── 6. Spreads vs CETES 28D ───────────────────────────────────────────────
    def calcular_spreads(self) -> "CETESDataMining":
        pivot = self.df.pivot_table(index="fecha", columns="plazo_dias",
                                    values="tasa_rendimiento", aggfunc="mean")
        if 28 not in pivot.columns:
            self.df["spread_vs_28d"] = np.nan
            return self

        ref_28 = pivot[28]
        for plazo in [91, 182, 364, 728]:
            if plazo not in pivot.columns:
                continue
            mask = self.df["plazo_dias"] == plazo
            fechas = self.df.loc[mask, "fecha"]
            self.df.loc[mask, "spread_vs_28d"] = (
                self.df.loc[mask, "tasa_rendimiento"].values
                - ref_28.reindex(fechas).values
            )
        return self

    # ── 7. Exportar ───────────────────────────────────────────────────────────
    def exportar(self) -> "CETESDataMining":
        self.df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        logger.info(f"Vista limpia exportada: {OUTPUT_PATH} ({len(self.df)} filas)")

        # Wide format (pivot) para Power BI alternativo
        wide = self.df.pivot_table(
            index="fecha", columns="serie_estandar",
            values=["tasa_rendimiento", "cambio_bp", "media_movil_4s", "spread_vs_28d"],
            aggfunc="mean",
        )
        wide.columns = ["_".join(col).strip() for col in wide.columns]
        wide.reset_index().to_csv(OUTPUT_WIDE, index=False, encoding="utf-8")
        logger.info(f"Vista wide exportada: {OUTPUT_WIDE}")

        # Registrar en auditoría
        self._registrar_auditoria("CLEAN_EXPORT", len(self.df),
                                  f"Exportados {OUTPUT_PATH} y {OUTPUT_WIDE}")
        return self

    def _registrar_auditoria(self, etapa: str, registros: int, detalle: str) -> None:
        """Registra la ejecución en la tabla auditoria_pipeline."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO auditoria_pipeline (etapa, registros_afectados, detalle)
                    VALUES (?, ?, ?)
                ''', (etapa, registros, detalle))
                conn.commit()
        except Exception as e:
            logger.warning(f"No se pudo registrar auditoría: {e}")

    # ── 8. Informe de calidad ─────────────────────────────────────────────────
    def informe_calidad(self) -> None:
        print("\n" + "="*60)
        print("INFORME DE CALIDAD DE DATOS")
        print("="*60)
        print(f"Período:      {self.df['fecha'].min().date()} → {self.df['fecha'].max().date()}")
        print(f"Registros:    {len(self.df)}")
        print(f"Plazos:       {sorted(self.df['plazo_dias'].unique().tolist())}")
        print(f"Imputados:    {self.df['dato_imputado'].sum()}")
        print(f"Outliers:     {self.df['es_outlier'].sum()}")
        print()
        print(self.df.groupby("plazo_dias")["tasa_rendimiento"].agg(
            Promedio="mean", Min="min", Max="max", N="count"
        ).round(4).to_string())
        print("="*60 + "\n")

    # ── Pipeline completo ─────────────────────────────────────────────────────
    def ejecutar(self) -> pd.DataFrame:
        self.cargar_datos()
        if self.df.empty:
            logger.error("No se encontraron datos en la base de datos. Abortando pipeline.")
            return self.df

        self.estandarizar()
        self.tratar_faltantes()
        self.calcular_derivados()
        self.detectar_outliers()


        for col in ["plazo_dias", "fecha", "tasa_rendimiento"]:
            if col not in self.df.columns:
                logger.error(f"Columna crítica '{col}' faltante después de detectar_outliers")
                logger.error(f"Columnas actuales: {list(self.df.columns)}")
                raise KeyError(f"Columna '{col}' perdida en el pipeline")

        self.calcular_spreads()
        self.exportar()
        return self.df


def main() -> None:
    pipeline = CETESDataMining()
    df = pipeline.ejecutar()
    pipeline.informe_calidad()


if __name__ == "__main__":
    main()