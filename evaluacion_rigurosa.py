"""Evaluación rigurosa del sistema con un archivo etiquetado.

Uso:
    python evaluacion_rigurosa.py "Dataset_Contraloria (1).xlsx"
    python evaluacion_rigurosa.py datos.xlsx --similitud 0.60 --salida evaluacion.xlsx
    python evaluacion_rigurosa.py datos.xlsx --referencia CONOSCE_1.xlsx CONOSCE_2.xlsx

Complementa a ``evaluar_dataset.py`` con lo que una evaluación necesita para
ser fiable:

- Intervalos de confianza al 95 % remuestreando ENTIDADES completas: las
  compras de una misma entidad no son independientes entre sí.
- Validación anidada dejando fuera una entidad cada vez: el umbral de
  similitud se elige con las demás entidades, sin ver la evaluada. Es la cifra
  que estima el rendimiento con entidades nuevas; la de la configuración
  ajustada con estos mismos datos es optimista.
- Comparación con líneas base (marcar todo, solo reglas, Isolation Forest
  por sí solo) y efecto de cada parámetro por separado. Con ``--referencia``
  se agrega el Isolation Forest entrenado con compras de referencia sin
  etiqueta (p. ej., CONOSCE) y aplicado a este archivo sin reentrenarse.
- Medidas complementarias, que responden a otra pregunta y no sustituyen a
  la medida por compra: detección por entidad y sensibilidad sobre compras de
  entidades con más de una compra.

La etiqueta NO interviene en el puntaje: solo se usa para medir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

from evaluar_dataset import _es_resumen, _leer, normalizar_y_emparejar_columnas, puntuar
from modulos_ia.modelo_aislamiento import (
    VARIABLES_IF_SOLO as VARIABLES_IF_BASE,
    ajustar_isolation_forest_solo,
    entrenar_isolation_forest_referencia,
    variables_isolation_forest_solo as variables_if_base,
)
from utilidades.gestor_modelo_referencia import preparar_referencia

OPCIONES_SIMILITUD = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
SIMILITUD_POR_DEFECTO = 0.50


def metricas(y: np.ndarray, decision: np.ndarray, puntaje: np.ndarray) -> dict:
    """Matriz de confusión y métricas adecuadas a clases desbalanceadas."""
    decision = np.asarray(decision, dtype=bool)
    vp = int((decision & (y == 1)).sum())
    fp = int((decision & (y == 0)).sum())
    fn = int((~decision & (y == 1)).sum())
    vn = int((~decision & (y == 0)).sum())
    return {
        "VP": vp, "FP": fp, "FN": fn, "VN": vn,
        "sensibilidad": vp / (vp + fn) if vp + fn else np.nan,
        "especificidad": vn / (vn + fp) if vn + fp else np.nan,
        "precision": vp / (vp + fp) if vp + fp else np.nan,
        "exactitud_balanceada": balanced_accuracy_score(y, decision),
        "mcc": matthews_corrcoef(y, decision),
        "roc_auc": roc_auc_score(y, puntaje) if len(set(y)) > 1 else np.nan,
    }


def intervalo_95(
    y: np.ndarray,
    decision: np.ndarray,
    entidades: np.ndarray,
    medida: str,
    repeticiones: int = 2000,
) -> tuple[float, float]:
    """Percentiles 2,5 y 97,5 de la medida al remuestrear entidades completas."""
    rng = np.random.default_rng(42)
    unicas = np.unique(entidades)
    filas = {e: np.flatnonzero(entidades == e) for e in unicas}
    decision = np.asarray(decision, dtype=bool)
    valores = []
    for _ in range(repeticiones):
        idx = np.concatenate([filas[e] for e in rng.choice(unicas, len(unicas))])
        yy, dd = y[idx], decision[idx]
        if len(set(yy)) < 2:
            continue
        if medida == "exactitud_balanceada":
            valores.append(balanced_accuracy_score(yy, dd))
        elif medida == "sensibilidad":
            valores.append(dd[yy == 1].mean())
        elif medida == "especificidad":
            valores.append(1 - dd[yy == 0].mean())
        else:
            raise ValueError(f"Medida sin intervalo: {medida}")
    if not valores:
        return np.nan, np.nan
    return tuple(float(v) for v in np.percentile(valores, [2.5, 97.5]))


def isolation_forest_solo(
    datos: pd.DataFrame, contaminacion: float, random_state: int = 42, referencia: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Modelo A: Isolation Forest por sí solo, sin NLP, reglas ni recurrencia.

    Es el mismo cálculo que la app usa al procesar con el modelo A. Recibe el
    archivo tal como se cargó (no la salida del extractor). Marca como
    fraccionamiento lo que el bosque considera atípico; el puntaje (mayor =
    más atípico) sirve para el ROC-AUC. La contaminación se fija de antemano,
    igual que en el sistema, para no ajustarla con la etiqueta.

    Con ``referencia`` el bosque se entrena con esas compras (sin etiqueta) y
    solo puntúa ``datos``; sin ella, se entrena con el propio archivo.
    """
    modelo = (None if referencia is None
              else entrenar_isolation_forest_referencia(referencia, contaminacion, random_state))
    return ajustar_isolation_forest_solo(datos, contaminacion, random_state, modelo)


def mismas_alertas(puntaje: np.ndarray, alertas: int) -> np.ndarray:
    """Marca las ``alertas`` compras de mayor puntaje (empates: la primera en el archivo)."""
    decision = np.zeros(len(puntaje), dtype=bool)
    decision[np.argsort(-np.asarray(puntaje, dtype=float), kind="stable")[:alertas]] = True
    return decision


def validacion_anidada(
    y: np.ndarray,
    entidades: np.ndarray,
    puntajes: dict[float, np.ndarray],
    umbral: float,
) -> tuple[np.ndarray, np.ndarray, dict[float, int]]:
    """Deja fuera cada entidad y elige la similitud solo con las demás.

    Empates: gana la similitud más cercana al valor por defecto, para no mover
    el parámetro sin evidencia.
    """
    decision = np.zeros(len(y), dtype=bool)
    puntaje = np.zeros(len(y))
    elegidas: dict[float, int] = {}
    for entidad in np.unique(entidades):
        dentro = entidades != entidad
        if len(set(y[dentro])) < 2:
            similitud = SIMILITUD_POR_DEFECTO
        else:
            similitud = max(
                puntajes,
                key=lambda u: (
                    balanced_accuracy_score(y[dentro], puntajes[u][dentro] >= umbral),
                    -abs(u - SIMILITUD_POR_DEFECTO),
                ),
            )
        elegidas[similitud] = elegidas.get(similitud, 0) + 1
        decision[~dentro] = puntajes[similitud][~dentro] >= umbral
        puntaje[~dentro] = puntajes[similitud][~dentro]
    return decision, puntaje, elegidas


def comparar(
    y: np.ndarray,
    entidades: np.ndarray,
    puntajes: dict[float, np.ndarray],
    reglas: np.ndarray,
    similitud: float,
    umbral: float,
    repeticiones: int = 2000,
    isolation_forest: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[pd.DataFrame, dict[float, int]]:
    """Tabla comparativa: líneas base, configuración elegida y validación anidada.

    ``isolation_forest`` asocia a cada variante del modelo A (p. ej.,
    "entrenado con este archivo") su decisión y su puntaje.
    """
    anidada, puntaje_anidado, elegidas = validacion_anidada(y, entidades, puntajes, umbral)
    candidatos = [
        ("Marcar todo como fraccionamiento", np.ones(len(y), bool), np.ones(len(y)), False),
        ("Solo reglas normativas", reglas >= umbral, reglas, True),
    ]
    # Con su contaminación el bosque solo puede marcar unos pocos casos; con
    # el mismo número de alertas que el híbrido se comparan en igualdad.
    alertas = int((puntajes[similitud] >= umbral).sum())
    for variante, (decision_if, puntaje_if) in (isolation_forest or {}).items():
        candidatos.append((f"Isolation Forest solo, {variante}", decision_if, puntaje_if, True))
        candidatos.append((f"Isolation Forest solo, {variante}, mismas alertas que el híbrido "
                           f"{similitud:.2f} ({alertas})",
                           mismas_alertas(puntaje_if, alertas), puntaje_if, True))
    candidatos += [
        (f"Híbrido, similitud {SIMILITUD_POR_DEFECTO:.2f} (por defecto)",
         puntajes[SIMILITUD_POR_DEFECTO] >= umbral, puntajes[SIMILITUD_POR_DEFECTO], True),
        (f"Híbrido, similitud {similitud:.2f} (ajustada a esta muestra)",
         puntajes[similitud] >= umbral, puntajes[similitud], True),
        ("Híbrido, validación anidada por entidad", anidada, puntaje_anidado, True),
    ]
    filas = []
    for nombre, decision, puntaje, con_intervalo in candidatos:
        fila = {"modelo": nombre, **metricas(y, decision, puntaje)}
        for medida in ("exactitud_balanceada", "sensibilidad", "especificidad"):
            bajo, alto = (
                intervalo_95(y, decision, entidades, medida, repeticiones)
                if con_intervalo else (np.nan, np.nan)
            )
            fila[f"{medida}_ic95_bajo"], fila[f"{medida}_ic95_alto"] = bajo, alto
        filas.append(fila)
    return pd.DataFrame(filas), elegidas


def triaje(y: np.ndarray, puntaje: np.ndarray, umbral: float, umbral_medio: float = 41.0) -> pd.DataFrame:
    """Casos por nivel de riesgo y propuesta de la ingesta."""
    tramos = [
        ("Alto", "Propuesta de confirmar", puntaje >= umbral),
        ("Medio", "Pendiente de revisión", (puntaje >= umbral_medio) & (puntaje < umbral)),
        ("Bajo", "Propuesta de descartar", puntaje < umbral_medio),
    ]
    return pd.DataFrame([
        {"nivel": nivel, "propuesta": propuesta,
         "fraccionamiento": int((m & (y == 1)).sum()), "conformes": int((m & (y == 0)).sum())}
        for nivel, propuesta, m in tramos
    ])


def composicion_etiqueta(y: np.ndarray, entidades: np.ndarray) -> pd.DataFrame:
    """Cuántas entidades tienen todas sus compras con la misma etiqueta.

    Si casi todas son homogéneas, la etiqueta describe la entidad auditada y
    no cada compra: es la clave para interpretar la sensibilidad por compra.
    """
    tabla = pd.DataFrame({"entidad": entidades, "y": y}).groupby("entidad")["y"].agg(["size", "sum"])
    tipo = np.where(tabla["sum"] == tabla["size"], "Todas con fraccionamiento",
                    np.where(tabla["sum"] == 0, "Todas conformes", "Mixta"))
    return (pd.Series(tipo).value_counts()
            .reindex(["Todas con fraccionamiento", "Todas conformes", "Mixta"], fill_value=0)
            .rename_axis("composicion").reset_index(name="entidades"))


def por_entidad(
    y: np.ndarray, entidades: np.ndarray, puntaje: np.ndarray, umbral: float, umbral_medio: float = 41.0
) -> pd.DataFrame:
    """Detección por entidad (medida complementaria).

    Una entidad cuenta como positiva si tiene alguna compra con
    fraccionamiento, y como detectada si alguna de sus compras alcanza el
    nivel indicado. Responde a otra pregunta que la medida por compra: si el
    sistema señala a las entidades auditadas.
    """
    t = pd.DataFrame({"entidad": entidades, "y": y, "alto": puntaje >= umbral, "medio": puntaje >= umbral_medio})
    g = t.groupby("entidad").agg(positiva=("y", "max"), alto=("alto", "any"), medio=("medio", "any"))
    filas = []
    for criterio, columna in (("Alguna compra en riesgo alto", "alto"),
                              ("Alguna compra en riesgo alto o medio", "medio")):
        positivas, negativas = g[g["positiva"] == 1], g[g["positiva"] == 0]
        filas.append({
            "criterio": criterio,
            "entidades_con_fraccionamiento": len(positivas),
            "detectadas": int(positivas[columna].sum()),
            "entidades_conformes": len(negativas),
            "conformes_con_alerta": int(negativas[columna].sum()),
            "sensibilidad": positivas[columna].mean() if len(positivas) else np.nan,
            "especificidad": 1 - negativas[columna].mean() if len(negativas) else np.nan,
        })
    return pd.DataFrame(filas)


def con_contexto(
    y: np.ndarray, entidades: np.ndarray, puntaje: np.ndarray, umbral: float, umbral_medio: float = 41.0
) -> pd.DataFrame:
    """Sensibilidad por compra según el contexto de la entidad (medida complementaria).

    Una compra única de su entidad no puede mostrar fraccionamiento por sí
    sola: no hay otra compra con la que se haya dividido.
    """
    compras = pd.Series(entidades).map(pd.Series(entidades).value_counts()).to_numpy()
    filas = []
    for grupo, mascara in (("Todas las compras", np.ones(len(y), bool)),
                           ("Entidades con más de una compra", compras > 1),
                           ("Entidades con una sola compra", compras == 1)):
        positivos = mascara & (y == 1)
        filas.append({
            "grupo": grupo,
            "fraccionamientos": int(positivos.sum()),
            "detectados_alto": int((positivos & (puntaje >= umbral)).sum()),
            "alto_o_pendiente": int((positivos & (puntaje >= umbral_medio)).sum()),
            "sensibilidad_alto": (positivos & (puntaje >= umbral)).sum() / positivos.sum() if positivos.any() else np.nan,
            "sensibilidad_alto_o_pendiente": (
                (positivos & (puntaje >= umbral_medio)).sum() / positivos.sum() if positivos.any() else np.nan),
        })
    return pd.DataFrame(filas)


def efecto_parametros(datos: pd.DataFrame, y: np.ndarray, base: dict) -> pd.DataFrame:
    """Mueve un parámetro cada vez, dejando el resto como en ``base``."""
    barridos = {
        "ventana": (15, 30, 45, 60, 90),
        "similitud": (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80),
        "contaminacion": (0.01, 0.03, 0.05, 0.10, 0.15),
        "umbral": (50, 55, 60, 65, 71, 75, 80),
    }
    cache: dict[tuple, np.ndarray] = {}
    filas = []
    for parametro, valores in barridos.items():
        for valor in valores:
            p = {**base, parametro: valor}
            clave = (p["ventana"], p["similitud"], p["contaminacion"])
            if clave not in cache:
                cache[clave] = puntuar(datos, p["ventana"], base["uit"], p["similitud"], p["contaminacion"])[
                    "puntaje_riesgo"].to_numpy(float)
            m = metricas(y, cache[clave] >= p["umbral"], cache[clave])
            filas.append({"parametro": parametro, "valor": valor, **m})
    return pd.DataFrame(filas)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivo", type=Path, help="Excel o CSV con la etiqueta de fraccionamiento")
    parser.add_argument("--ventana", type=int, default=30, help="Ventana de recurrencia en días (30)")
    parser.add_argument("--similitud", type=float, default=0.60, help="Umbral de similitud evaluado (0.60)")
    parser.add_argument("--umbral", type=float, default=71.0, help="Puntaje de riesgo Alto (71)")
    parser.add_argument("--contaminacion", type=float, default=0.03, help="Proporción atípica esperada (0.03)")
    parser.add_argument("--uit", type=float, default=5150.0, help="UIT para años sin valor oficial (5150)")
    parser.add_argument("--repeticiones", type=int, default=2000, help="Remuestreos para los intervalos (2000)")
    parser.add_argument("--salida", type=Path, help="Excel donde guardar las tablas")
    parser.add_argument("--referencia", type=Path, nargs="+",
                        help="Excel o CSV sin etiqueta (p. ej., CONOSCE) para entrenar el Isolation Forest solo")
    args = parser.parse_args(argv)

    for ruta in (args.archivo, *(args.referencia or ())):
        if not ruta.exists():
            print(f"No existe el archivo: {ruta}", file=sys.stderr)
            return 1
    opciones = tuple(sorted(set(OPCIONES_SIMILITUD) | {args.similitud}))

    datos = normalizar_y_emparejar_columnas(_leer(args.archivo))
    if "es_fraccionamiento" not in datos.columns:
        print("El archivo no tiene una columna de etiqueta (es_fraccionamiento / label).", file=sys.stderr)
        return 1
    datos = datos.loc[~_es_resumen(datos)].reset_index(drop=True)
    etiqueta = pd.to_numeric(datos["es_fraccionamiento"], errors="coerce")
    datos = datos.loc[etiqueta.isin([0, 1])].reset_index(drop=True)
    y = etiqueta.loc[etiqueta.isin([0, 1])].astype(int).to_numpy()
    entidades = datos["entidad_ruc"].astype(str).to_numpy()

    resultados = {
        u: puntuar(datos, args.ventana, args.uit, u, args.contaminacion) for u in opciones
    }
    puntajes = {u: r["puntaje_riesgo"].to_numpy(float) for u, r in resultados.items()}
    reglas = resultados[SIMILITUD_POR_DEFECTO]["score_reglas_negocio"].to_numpy(float)

    resumen = pd.DataFrame([{
        "compras_evaluadas": len(y), "con_fraccionamiento": int(y.sum()), "conformes": int((y == 0).sum()),
        "entidades": len(set(entidades)), "entidades_con_conformes": len(set(entidades[y == 0])),
        "ventana": args.ventana, "similitud": args.similitud, "umbral": args.umbral,
        "contaminacion": args.contaminacion, "repeticiones_ic": args.repeticiones,
    }])
    variantes_if = {"entrenado con este archivo": isolation_forest_solo(datos, args.contaminacion)}
    if args.referencia:
        referencia = preparar_referencia([_leer(r) for r in args.referencia])
        resumen["compras_referencia_if"] = len(referencia)
        variantes_if[f"entrenado con referencia ({len(referencia):,} compras)"] = isolation_forest_solo(
            datos, args.contaminacion, referencia=referencia)
    comparacion, elegidas = comparar(
        y, entidades, puntajes, reglas, args.similitud, args.umbral, args.repeticiones, variantes_if)
    tabla_triaje = triaje(y, puntajes[args.similitud], args.umbral)
    composicion = composicion_etiqueta(y, entidades)
    tabla_entidad = por_entidad(y, entidades, puntajes[args.similitud], args.umbral)
    tabla_contexto = con_contexto(y, entidades, puntajes[args.similitud], args.umbral)
    parametros = efecto_parametros(datos, y, {
        "ventana": args.ventana, "similitud": args.similitud, "contaminacion": args.contaminacion,
        "umbral": args.umbral, "uit": args.uit,
    })

    pd.set_option("display.width", 220)
    print(resumen.T.to_string(header=False))
    print("\nComparación (IC 95 % remuestreando entidades):")
    print(comparacion.round(3).to_string(index=False))
    print("\nSimilitud elegida en la validación anidada (veces):", elegidas)
    print("\nTriaje con la configuración evaluada:")
    print(tabla_triaje.to_string(index=False))
    print("\nMedidas complementarias (responden a otra pregunta; no sustituyen a la medida por compra)")
    print("Composición de la etiqueta por entidad:")
    print(composicion.to_string(index=False))
    print("\nDetección por entidad:")
    print(tabla_entidad.round(3).to_string(index=False))
    print("\nSensibilidad por compra según el contexto de la entidad:")
    print(tabla_contexto.round(3).to_string(index=False))
    print("\nEfecto de cada parámetro (el resto como en la configuración evaluada):")
    print(parametros[["parametro", "valor", "VP", "FP", "FN", "VN", "exactitud_balanceada", "roc_auc"]]
          .round(3).to_string(index=False))

    if args.salida:
        with pd.ExcelWriter(args.salida) as escritor:
            resumen.to_excel(escritor, sheet_name="Datos", index=False)
            comparacion.to_excel(escritor, sheet_name="Comparacion", index=False)
            pd.DataFrame([{"similitud": u, "veces_elegida": n} for u, n in sorted(elegidas.items())]).to_excel(
                escritor, sheet_name="Anidada", index=False)
            tabla_triaje.to_excel(escritor, sheet_name="Triaje", index=False)
            composicion.to_excel(escritor, sheet_name="Etiqueta", index=False)
            tabla_entidad.to_excel(escritor, sheet_name="PorEntidad", index=False)
            tabla_contexto.to_excel(escritor, sheet_name="ConContexto", index=False)
            parametros.to_excel(escritor, sheet_name="Parametros", index=False)
        print(f"\nTablas guardadas en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
