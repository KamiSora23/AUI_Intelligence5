"""Persistencia del modelo A entrenado con compras de referencia (p. ej., CONOSCE).

Uso desde la línea de comandos (entrena y guarda):
    python -m utilidades.gestor_modelo_referencia CONOSCE_1.xlsx CONOSCE_2.xlsx
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from modulos_ia.modelo_aislamiento import entrenar_isolation_forest_referencia
from utilidades.mapeador_columnas import normalizar_y_emparejar_columnas

CARPETA_MODELOS = Path(__file__).resolve().parents[1] / "modelos_entrenados"
RUTA_MODELO = CARPETA_MODELOS / "isolation_forest_referencia.joblib"


def preparar_referencia(tablas: list[pd.DataFrame]) -> pd.DataFrame:
    """Normaliza y une las tablas; quedan solo las compras con monto mayor a 0."""
    referencia = pd.concat([normalizar_y_emparejar_columnas(t) for t in tablas], ignore_index=True)
    monto = pd.to_numeric(referencia["monto_referencial"], errors="coerce").fillna(0.0)
    return referencia.loc[monto > 0].reset_index(drop=True)


def entrenar_y_guardar(
    tablas: list[pd.DataFrame], archivos: list[str], contaminacion: float = 0.03
) -> dict:
    """Entrena el modelo A con las compras de referencia y lo deja guardado."""
    referencia = preparar_referencia(tablas)
    guardado = {
        "modelo": entrenar_isolation_forest_referencia(referencia, contaminacion),
        "compras": len(referencia),
        "entidades": int(referencia["entidad_ruc"].astype(str).nunique()),
        "archivos": list(archivos),
        "contaminacion": float(contaminacion),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    CARPETA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(guardado, RUTA_MODELO)
    return guardado


def modelo_referencia() -> dict | None:
    """Modelo guardado con sus datos de entrenamiento, o ``None`` si no hay."""
    if not RUTA_MODELO.exists():
        return None
    return joblib.load(RUTA_MODELO)


def eliminar() -> None:
    RUTA_MODELO.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivos", type=Path, nargs="+", help="Excel o CSV con compras de referencia")
    parser.add_argument("--contaminacion", type=float, default=0.03, help="Proporción atípica esperada (0.03)")
    args = parser.parse_args(argv)
    tablas = [pd.read_csv(r) if r.suffix.lower() == ".csv" else pd.read_excel(r) for r in args.archivos]
    guardado = entrenar_y_guardar(tablas, [r.name for r in args.archivos], args.contaminacion)
    print(f"Modelo A entrenado con {guardado['compras']:,} compras de {guardado['entidades']:,} entidades "
          f"(contaminación {guardado['contaminacion']}) y guardado en {RUTA_MODELO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
