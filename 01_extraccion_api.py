from dotenv import load_dotenv
import os
load_dotenv()

#!/usr/bin/env python3
"""
================================================================================
ETAPA 1: EXTRACCIÓN vía API SIE BANXICO 
AFORE XXI Banorte - Prueba Técnica Analista de Data Science
================================================================================
"""

import requests
import json
import logging
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s | %(message)s',
    handlers=[
        logging.FileHandler('pipeline_cetes.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SIE.Extraccion")


class SIEApiClient:
    """
    Cliente para la API REST del SIE de Banco de México.

    Implementa: autenticación Bmx-Token, manejo de errores HTTP,
    backoff exponencial con jitter, rate limiting, logging y
    guardado de checkpoint incremental.
    """

    BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1"
    MAX_SERIES_PER_REQUEST = 20
    BASE_DELAY = 1.0          # segundos mínimo entre requests
    MAX_RETRIES = 4

    SERIES_CETES: Dict[str, Dict] = {
        "SF43936":  {"plazo_dias": 28,  "frecuencia": "semanal",   "nombre": "CETES 28 días"},
        "SF43939":  {"plazo_dias": 91,  "frecuencia": "semanal",   "nombre": "CETES 91 días"},
        "SF43942":  {"plazo_dias": 182, "frecuencia": "semanal",   "nombre": "CETES 182 días"},
        "SF43945":  {"plazo_dias": 364, "frecuencia": "semanal",   "nombre": "CETES 364 días"},
        "SF349785": {"plazo_dias": 728, "frecuencia": "quincenal", "nombre": "CETES 728 días"},
    }

    def __init__(self, token: str, output_dir: str = "."):
        self.token = token
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "Bmx-Token": token,
            "Accept": "application/json",
        })
        self._validate_token()
        logger.info("SIEApiClient inicializado y token validado.")

    def _validate_token(self) -> None:
        """Valida el token haciendo una petición mínima antes de iniciar."""
        url = f"{self.BASE_URL}/series/SF43936/datos/2026-01-01/2026-01-07"
        r = self.session.get(url, timeout=15)
        if r.status_code == 400:
            raise ValueError(f"Token inválido o expirado: {r.text[:200]}")
        if r.status_code not in (200, 404):
            logger.warning(f"Validación de token retornó HTTP {r.status_code}")

    def _backoff(self, attempt: int) -> float:
        """Calcula espera con jitter: 2^attempt + jitter aleatorio 0-1s."""
        return 2 ** attempt + random.uniform(0, 1)

    def _make_request(self, endpoint: str) -> Optional[Dict]:
        url = f"{self.BASE_URL}/{endpoint}"
        for attempt in range(self.MAX_RETRIES):
            try:
                time.sleep(self.BASE_DELAY)
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 400:
                    raise ValueError(f"Error 400 — Token o parámetros inválidos: {resp.text[:200]}")
                if resp.status_code == 429:
                    wait = self._backoff(attempt + 1)
                    logger.warning(f"Rate limit (429). Esperando {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout intento {attempt+1}/{self.MAX_RETRIES}")
                time.sleep(self._backoff(attempt))
            except requests.exceptions.ConnectionError as e:
                logger.error(f"ConnectionError: {e}")
                time.sleep(self._backoff(attempt))
        logger.error(f"Máximo de reintentos alcanzado para: {url}")
        return None

    def _parse_series(self, data: Dict) -> List[Dict]:
        """Parsea la respuesta de la API a una lista de registros planos."""
        registros = []
        for serie in data.get("bmx", {}).get("series", []):
            sid = serie.get("idSerie", "")
            cfg = self.SERIES_CETES.get(sid, {})
            for dato in serie.get("datos", []):
                raw_val = dato.get("dato", "N/E")
                registros.append({
                    "serie_id":        sid,
                    "plazo_dias":      cfg.get("plazo_dias"),
                    "nombre_serie":    cfg.get("nombre", serie.get("titulo", "")),
                    "fecha":           dato.get("fecha"),        # DD/MM/YYYY
                    "dato_raw":        raw_val,
                    "es_ne":           raw_val == "N/E",
                    "frecuencia":      cfg.get("frecuencia", ""),
                    "unidad":          "% anual",
                    "fecha_extraccion": datetime.now().isoformat(),
                })
        return registros

    def obtener_series_periodo(
        self,
        serie_ids: List[str],
        fecha_inicio: str,
        fecha_fin: str,
    ) -> List[Dict]:
        """Consulta series en batches de MAX_SERIES_PER_REQUEST."""
        all_registros: List[Dict] = []
        for i in range(0, len(serie_ids), self.MAX_SERIES_PER_REQUEST):
            batch = serie_ids[i: i + self.MAX_SERIES_PER_REQUEST]
            ids_str = ",".join(batch)
            endpoint = f"series/{ids_str}/datos/{fecha_inicio}/{fecha_fin}"
            data = self._make_request(endpoint)
            if data:
                registros = self._parse_series(data)
                all_registros.extend(registros)
                logger.info(f"Batch {i//self.MAX_SERIES_PER_REQUEST+1}: {len(registros)} registros extraídos")
        return all_registros

    def obtener_ultimos_12_meses(self) -> pd.DataFrame:
        """Extrae CETES de los últimos 12 meses y devuelve DataFrame."""
        fecha_fin    = datetime.now().strftime("%Y-%m-%d")
        fecha_inicio = (datetime.now() - timedelta(days=366)).strftime("%Y-%m-%d")
        logger.info(f"Período de consulta: {fecha_inicio} → {fecha_fin}")

        registros = self.obtener_series_periodo(
            list(self.SERIES_CETES.keys()),
            fecha_inicio,
            fecha_fin,
        )
        df = pd.DataFrame(registros)
        self._log_resumen(df)
        return df

    def _log_resumen(self, df: pd.DataFrame) -> None:
        total = len(df)
        ne_pct = df["es_ne"].mean() * 100
        logger.info(f"Total registros: {total} | N/E: {ne_pct:.1f}%")
        for plazo, g in df.groupby("plazo_dias"):
            pct = g["es_ne"].mean() * 100
            logger.info(f"  CETES {plazo:>3}D → {len(g)} registros | {pct:.1f}% N/E")


def main() -> None:
    TOKEN = os.getenv("BANXICO_TOKEN")
    client = SIEApiClient(TOKEN)
    df = client.obtener_ultimos_12_meses()
    path = "datos_cetes_raw.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Guardado: {path} ({len(df)} registros)")


if __name__ == "__main__":
    main()
