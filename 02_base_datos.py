from dotenv import load_dotenv
import os
load_dotenv()

#!/usr/bin/env python3
"""
================================================================================
ETAPA 2: BASE DE DATOS - SQLite (PostgreSQL/MySQL compatible)
AFORE XXI Banorte - Prueba Técnica Analista de Data Science
================================================================================
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict
from pathlib import Path

DB_PATH = "cetes_database.db"
CSV_PATH = "datos_cetes_raw.csv"


class CETESDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cetes_rendimientos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    serie_id TEXT NOT NULL,
                    plazo_dias INTEGER NOT NULL,
                    nombre_serie TEXT NOT NULL,
                    fecha DATE NOT NULL,
                    tasa_rendimiento REAL,
                    es_dato_faltante BOOLEAN DEFAULT 0,
                    frecuencia TEXT NOT NULL,
                    unidad TEXT NOT NULL,
                    fecha_extraccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(serie_id, fecha)
                )
            ''')

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plazo ON cetes_rendimientos(plazo_dias)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fecha ON cetes_rendimientos(fecha)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plazo_fecha ON cetes_rendimientos(plazo_dias, fecha)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_serie ON cetes_rendimientos(serie_id)")

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auditoria_pipeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    etapa TEXT NOT NULL,
                    registros_afectados INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    detalle TEXT
                )
            ''')

            conn.commit()

    def insertar_registros(self, registros: List[Dict]) -> int:
        insertados = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for reg in registros:

                dato_raw = reg.get("dato_raw", "")
                

                es_ne = (dato_raw == "N/E")
                

                if not es_ne:
                    es_ne_val = reg.get("es_ne", False)
                    if isinstance(es_ne_val, str):
                        es_ne = es_ne_val.lower() in ("true", "1", "yes", "si")
                    else:
                        es_ne = bool(es_ne_val)
                

                if es_ne or dato_raw == "" or dato_raw is None:
                    tasa = None
                else:
                    try:
                        tasa = float(dato_raw)
                    except (ValueError, TypeError):
                        tasa = None
                        es_ne = True  # Si no se puede convertir, marcar como faltante

                # Convertir fecha de DD/MM/YYYY a YYYY-MM-DD
                fecha_raw = reg.get("fecha", "")
                try:
                    fecha_dt = datetime.strptime(fecha_raw, "%d/%m/%Y")
                    fecha_iso = fecha_dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    fecha_iso = fecha_raw

                cursor.execute('''
                    INSERT OR REPLACE INTO cetes_rendimientos 
                    (serie_id, plazo_dias, nombre_serie, fecha, tasa_rendimiento, 
                     es_dato_faltante, frecuencia, unidad, fecha_extraccion)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    reg.get("serie_id"),
                    reg.get("plazo_dias"),
                    reg.get("nombre_serie"),
                    fecha_iso,
                    tasa,
                    1 if es_ne else 0,  #Asegurar que sea 0 o 1
                    reg.get("frecuencia"),
                    reg.get("unidad"),
                    reg.get("fecha_extraccion", datetime.now().isoformat())
                ))
                insertados += 1

            conn.commit()

            cursor.execute('''
                INSERT INTO auditoria_pipeline (etapa, registros_afectados, detalle)
                VALUES (?, ?, ?)
            ''', ("INSERT", insertados, f"Inserción masiva de {insertados} registros"))
            conn.commit()

        return insertados

    def cargar_desde_csv(self, csv_path: str = CSV_PATH) -> int:
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"No se encontró el archivo {csv_path}. "
                f"Ejecuta primero la Etapa 1 (01_extraccion_api_v2.py)."
            )

        df = pd.read_csv(csv_path, encoding="utf-8")
        if df.empty:
            raise ValueError(f"El archivo {csv_path} está vacío.")

        registros = df.to_dict("records")
        insertados = self.insertar_registros(registros)

        print(f" {insertados} registros cargados desde {csv_path} → {self.db_path}")
        return insertados

    def consultar_por_plazo_fecha(self, plazo: int, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
        query = '''
            SELECT * FROM cetes_rendimientos
            WHERE plazo_dias = ? AND fecha BETWEEN ? AND ?
            ORDER BY fecha
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(plazo, fecha_inicio, fecha_fin))

    def obtener_ultimo_snapshot(self) -> pd.DataFrame:
        query = '''
            SELECT cr.* FROM cetes_rendimientos cr
            INNER JOIN (
                SELECT plazo_dias, MAX(fecha) as max_fecha 
                FROM cetes_rendimientos 
                WHERE es_dato_faltante = 0
                GROUP BY plazo_dias
            ) latest ON cr.plazo_dias = latest.plazo_dias AND cr.fecha = latest.max_fecha
            ORDER BY cr.plazo_dias
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)


def main():
    db = CETESDatabase()
    print(f"Base de datos inicializada: {DB_PATH}")

    try:
        insertados = db.cargar_desde_csv(CSV_PATH)
        print(f"Tabla poblada con {insertados} registros.")
    except FileNotFoundError as e:
        print(f" {e}")
        print("Solo se creó la estructura. Ejecuta la Etapa 1 primero.")


if __name__ == "__main__":
    main()