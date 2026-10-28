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


def _claves(tabla: pd.DataFrame) -> pd.Series:
    monto = pd.to_numeric(tabla["monto_referencial"], errors="coerce").round(2)
    return pd.Series(list(zip(tabla["entidad_ruc"].astype(str), monto, tabla["fecha_convocatoria"].astype(str))),
                     index=tabla.index)


def preparar_referencia(tablas: list[pd.DataFrame]) -> pd.DataFrame:
    """Normaliza y une las tablas; quedan solo las compras con monto mayor a 0.

    Un mismo ítem puede venir en varios archivos (p. ej., las convocatorias y
    las adjudicaciones de 2024): se descarta la compra cuya entidad, monto y
    fecha ya estaban en un archivo anterior, para no contarla dos veces. Las
    repeticiones dentro de un mismo archivo se conservan: pueden ser ítems
    distintos de igual monto.
    """
    vistas: set = set()
    partes = []
    duplicadas = 0
    for tabla in tablas:
        normalizada = normalizar_y_emparejar_columnas(tabla)
        claves = _claves(normalizada)
        repetida = claves.isin(vistas)
        duplicadas += int(repetida.sum())
        partes.append(normalizada.loc[~repetida])
        vistas.update(claves)
    referencia = pd.concat(partes, ignore_index=True)
    monto = pd.to_numeric(referencia["monto_referencial"], errors="coerce").fillna(0.0)
    referencia = referencia.loc[monto > 0].reset_index(drop=True)
    referencia.attrs["duplicadas_entre_archivos"] = duplicadas
    return referencia


def entrenar_y_guardar(
    tablas: list[pd.DataFrame], archivos: list[str], contaminacion: float = 0.03
) -> dict:
    """Entrena el modelo A con las compras de referencia y lo deja guardado."""
    referencia = preparar_referencia(tablas)
    guardado = {
        "modelo": entrenar_isolation_forest_referencia(referencia, contaminacion),
        "compras": len(referencia),
        "duplicadas_entre_archivos": int(referencia.attrs.get("duplicadas_entre_archivos", 0)),
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
          f"(contaminación {guardado['contaminacion']}; {guardado['duplicadas_entre_archivos']:,} repetidas "
          f"entre archivos descartadas) y guardado en {RUTA_MODELO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
