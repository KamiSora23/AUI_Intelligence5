"""Persistencia de los Isolation Forest entrenados con compras de referencia (p. ej., del OECE).

Hay un modelo guardado por cada modelo de detección:

- Modelo A (Isolation Forest solo): variables de monto y de la entidad.
- Modelo B (sistema completo): el bosque que da el ajuste de 0 a 20 puntos,
  con las variables de recurrencia, similitud y monto acumulado. Como esas
  variables dependen de la ventana, la UIT y el umbral de similitud, se
  guardan junto al modelo.

Uso desde la línea de comandos (entrena y guarda):
    python -m utilidades.gestor_modelo_referencia CONOSCE_1.xlsx CONOSCE_2.xlsx
    python -m utilidades.gestor_modelo_referencia --modelo B CONOSCE_1.xlsx CONOSCE_2.xlsx
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from modulos_ia.extractor_caracteristicas import extraer_caracteristicas_temporales_y_nlp
from modulos_ia.modelo_aislamiento import (
    entrenar_isolation_forest_hibrido,
    entrenar_isolation_forest_referencia,
)
from utilidades.mapeador_columnas import normalizar_y_emparejar_columnas

CARPETA_MODELOS = Path(__file__).resolve().parents[1] / "modelos_entrenados"
RUTA_MODELO = CARPETA_MODELOS / "isolation_forest_referencia.joblib"
RUTA_MODELO_B = CARPETA_MODELOS / "isolation_forest_hibrido_referencia.joblib"


def _ruta(modelo: str) -> Path:
    return RUTA_MODELO_B if modelo == "B" else RUTA_MODELO


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


def entrenar_modelo_b(
    referencia: pd.DataFrame,
    contaminacion: float = 0.03,
    ventana_dias: int = 30,
    uit_vigente: float = 5150.0,
    umbral_similitud_nlp: float = 0.50,
):
    """Calcula las variables del modelo B en las compras de referencia y entrena su bosque."""
    caracteristicas = extraer_caracteristicas_temporales_y_nlp(
        referencia,
        ventana_dias=ventana_dias,
        uit_vigente=uit_vigente,
        umbral_similitud_nlp=umbral_similitud_nlp,
    )
    return entrenar_isolation_forest_hibrido(caracteristicas, contaminacion)


def entrenar_y_guardar(
    tablas: list[pd.DataFrame],
    archivos: list[str],
    contaminacion: float = 0.03,
    modelo: str = "A",
    ventana_dias: int = 30,
    uit_vigente: float = 5150.0,
    umbral_similitud_nlp: float = 0.50,
) -> dict:
    """Entrena el modelo indicado con las compras de referencia y lo deja guardado."""
    referencia = preparar_referencia(tablas)
    guardado = {
        "compras": len(referencia),
        "duplicadas_entre_archivos": int(referencia.attrs.get("duplicadas_entre_archivos", 0)),
        "entidades": int(referencia["entidad_ruc"].astype(str).nunique()),
        "archivos": list(archivos),
        "contaminacion": float(contaminacion),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if modelo == "B":
        guardado["modelo"] = entrenar_modelo_b(
            referencia, contaminacion, ventana_dias, uit_vigente, umbral_similitud_nlp)
        guardado.update({
            "ventana_dias": int(ventana_dias),
            "uit_vigente": float(uit_vigente),
            "umbral_similitud_nlp": float(umbral_similitud_nlp),
        })
    else:
        guardado["modelo"] = entrenar_isolation_forest_referencia(referencia, contaminacion)
    CARPETA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(guardado, _ruta(modelo))
    return guardado


def modelo_referencia(modelo: str = "A") -> dict | None:
    """Modelo guardado con sus datos de entrenamiento, o ``None`` si no hay."""
    ruta = _ruta(modelo)
    if not ruta.exists():
        return None
    return joblib.load(ruta)


def eliminar(modelo: str = "A") -> None:
    _ruta(modelo).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivos", type=Path, nargs="+", help="Excel o CSV con compras de referencia")
    parser.add_argument("--modelo", choices=("A", "B"), default="A", help="Modelo que se entrena (A)")
    parser.add_argument("--contaminacion", type=float, default=0.03, help="Proporción atípica esperada (0.03)")
    parser.add_argument("--ventana", type=int, default=30, help="Modelo B: ventana de recurrencia en días (30)")
    parser.add_argument("--similitud", type=float, default=0.50, help="Modelo B: umbral de similitud (0.50)")
    parser.add_argument("--uit", type=float, default=5150.0, help="Modelo B: UIT para años sin valor oficial (5150)")
    args = parser.parse_args(argv)
    tablas = [pd.read_csv(r) if r.suffix.lower() == ".csv" else pd.read_excel(r) for r in args.archivos]
    guardado = entrenar_y_guardar(tablas, [r.name for r in args.archivos], args.contaminacion, args.modelo,
                                  args.ventana, args.uit, args.similitud)
    print(f"Modelo {args.modelo} entrenado con {guardado['compras']:,} compras de {guardado['entidades']:,} "
          f"entidades (contaminación {guardado['contaminacion']}; {guardado['duplicadas_entre_archivos']:,} "
          f"repetidas entre archivos descartadas) y guardado en {_ruta(args.modelo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
