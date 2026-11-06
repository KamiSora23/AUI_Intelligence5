"""Evalúa el motor de detección contra un dataset con etiqueta de fraccionamiento.

Uso:
    python evaluar_dataset.py Dataset_Contraloria.xlsx
    python evaluar_dataset.py datos.xlsx --salida resultados.xlsx
    python evaluar_dataset.py datos.xlsx --contexto CONOSCE_CONVOCATORIAS_CD2024_0.xlsx
    python evaluar_dataset.py datos.xlsx --calibrar

El archivo debe incluir una columna 0/1 (``es_fraccionamiento`` o un alias como
``label``). La etiqueta NO se usa para puntuar: solo para medir el resultado.
Si el archivo no trae el tipo de proceso, se asume "Igual o Menor a 8 UIT".

Antes de medir, el script audita la etiqueta: avisa si hay filas que son
resúmenes de hallazgos (no compras), palabras de auditoría escritas en el texto,
clases muy desbalanceadas o entidades con una sola fila. Esos defectos inflan o
vuelven imposible la medición sin que el modelo tenga culpa.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from modulos_ia.evaluador_metricas import (
    UMBRAL_ALTO,
    barrido_de_umbrales,
    evaluar_contra_etiquetas,
    metricas_de_referencia,
)
from modulos_ia.extractor_caracteristicas import (
    _texto_basico,
    es_resumen_de_hallazgo,
    extraer_caracteristicas_temporales_y_nlp,
)
from modulos_ia.modelo_aislamiento import ejecutar_modelo_aislamiento_hibrido
from utilidades.mapeador_columnas import normalizar_y_emparejar_columnas

VENTANAS = (30, 60, 90, 180, 365)
UMBRALES_NLP = (0.30, 0.40, 0.50, 0.60)
UMBRALES_PUNTAJE = tuple(range(30, 85, 5))


def _leer(ruta: Path) -> pd.DataFrame:
    if Path(ruta).suffix.lower() == ".csv":
        return pd.read_csv(ruta)
    return pd.read_excel(ruta)


def _es_resumen(datos: pd.DataFrame) -> pd.Series:
    """Filas que describen un hallazgo en vez de una compra (ver el módulo del motor)."""
    return es_resumen_de_hallazgo(datos)


def auditar_etiquetas(datos: pd.DataFrame) -> None:
    et = pd.to_numeric(datos["es_fraccionamiento"], errors="coerce")
    pos, neg = et == 1, et == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        print("AUDITORÍA: el archivo solo tiene una clase; no se puede medir el desempeño.")
        return
    resumen = _es_resumen(datos)
    base = metricas_de_referencia(et[pos | neg])
    ent = datos["entidad_ruc"].astype(str)
    por_entidad = ent.value_counts()
    monto = pd.to_numeric(datos["monto_referencial"], errors="coerce").fillna(0.0)

    print("AUDITORÍA DE LA ETIQUETA")
    print(f"  Casos: {n_pos} con fraccionamiento ({n_pos / (n_pos + n_neg):.0%}) y {n_neg} conformes.")
    print(
        f"  Marcar TODO como fraccionamiento ya da precisión {base['precision']:.3f} y "
        f"F1 {base['f1_score']:.3f}; por eso se reportan también especificidad y exactitud balanceada."
    )
    if resumen.any():
        print(
            f"  {int(resumen.sum())} filas son resúmenes de hallazgo, no compras "
            f"({int((resumen & pos).sum())} positivas, {int((resumen & neg).sum())} conformes): "
            f"su texto trae una palabra de auditoría Y su CUBSO es 'no definido'."
        )
    sin_monto = int((neg & (monto <= 0)).sum())
    if sin_monto:
        print(f"  {sin_monto} conformes tienen monto 0: no son compras reales y no ponen a prueba al modelo.")
    print(
        f"  Entidades distintas: {ent[pos].nunique()} con casos positivos y {ent[neg].nunique()} con conformes."
    )
    print(
        f"  {(por_entidad == 1).mean():.0%} de las entidades tienen una sola fila: sin las demás compras "
        f"de la entidad no se puede ver recurrencia ni acumulado (use --contexto)."
    )


def agregar_contexto(datos: pd.DataFrame, rutas) -> pd.DataFrame:
    """Suma filas sin etiqueta de las mismas entidades y años.

    Un caso etiquetado suele venir con una sola fila; el fraccionamiento solo se
    ve junto a las demás compras de esa entidad. Las filas de contexto se usan
    para puntuar pero no entran en las métricas.
    """
    entidades = set(datos["entidad_ruc"].astype(str))
    anios = set(pd.to_datetime(datos["fecha_convocatoria"]).dt.year)
    partes = []
    for ruta in rutas:
        ctx = normalizar_y_emparejar_columnas(_leer(Path(ruta)))
        ctx = ctx[ctx["entidad_ruc"].astype(str).isin(entidades)]
        ctx = ctx[pd.to_datetime(ctx["fecha_convocatoria"]).dt.year.isin(anios)]
        partes.append(ctx)
    if not partes:
        return datos
    contexto = pd.concat(partes, ignore_index=True)

    def clave(d):
        return (
            d["entidad_ruc"].astype(str) + "|" + d["fecha_convocatoria"].astype(str) + "|"
            + pd.to_numeric(d["monto_referencial"], errors="coerce").round(2).astype(str) + "|"
            + d["descripcion_item"].astype(str).apply(_texto_basico)
        )

    contexto["_clave"] = clave(contexto)
    contexto = contexto[~contexto["_clave"].isin(set(clave(datos)))]
    contexto = contexto.drop_duplicates("_clave").drop(columns="_clave")
    contexto["es_fraccionamiento"] = np.nan
    contexto["es_contexto"] = True
    datos = datos.copy()
    datos["es_contexto"] = False
    print(f"Contexto: {len(contexto):,} compras sin etiqueta de las mismas entidades y años.")
    return pd.concat([datos, contexto], ignore_index=True)


def puntuar(datos, ventana, uit, umbral_nlp, contaminacion, modelo=None) -> pd.DataFrame:
    """Modelo B; con ``modelo`` (bosque entrenado con referencia) el bosque solo puntúa."""
    caracteristicas = extraer_caracteristicas_temporales_y_nlp(
        datos, ventana_dias=ventana, uit_vigente=uit, umbral_similitud_nlp=umbral_nlp,
    )
    return ejecutar_modelo_aislamiento_hibrido(
        caracteristicas, contaminacion=contaminacion, modo_automatico=False, modelo=modelo,
    )


def _mostrar(titulo: str, m: dict) -> None:
    def fmt(valor):
        return "N/D" if valor is None else f"{valor:.3f}"

    print(f"\n{titulo}")
    print(
        f"  Sensibilidad {fmt(m['recall'])} | Especificidad {fmt(m['especificidad'])} | "
        f"Exactitud balanceada {fmt(m['exactitud_balanceada'])} | MCC {fmt(m['mcc'])}"
    )
    print(
        f"  Precisión {fmt(m['precision'])} | F1 {fmt(m['f1_score'])} | "
        f"ROC-AUC {fmt(m['roc_auc'])} | PR-AUC {fmt(m['pr_auc'])}"
    )
    print(
        f"  VP {m['verdaderos_positivos']} | FP {m['falsos_positivos']} | "
        f"FN {m['falsos_negativos']} | VN {m['verdaderos_negativos']}"
    )


def _dividir_entidades(datos, proporcion, semilla):
    """Reparte las entidades etiquetadas entre entrenamiento y prueba.

    Las entidades (no las filas) se separan para no partir las compras de una
    misma entidad, y se hace por separado con las que tienen conformes y las que
    no, para que ambas partes tengan casos de las dos clases.
    """
    et = pd.to_numeric(datos["es_fraccionamiento"], errors="coerce")
    valida = et.isin([0, 1]).to_numpy()
    ent = datos["entidad_ruc"].astype(str)
    con_conformes = set(ent[et == 0])
    rng = np.random.default_rng(semilla)
    entrenamiento = set()
    for grupo in (sorted(set(ent[valida]) & con_conformes), sorted(set(ent[valida]) - con_conformes)):
        grupo = np.array(grupo)
        rng.shuffle(grupo)
        entrenamiento |= set(grupo[: int(round(len(grupo) * proporcion))])
    en_tr = ent.isin(entrenamiento).to_numpy()
    return en_tr & valida, ~en_tr & valida, len(con_conformes)


def calibrar(datos, uit, contaminacion, proporcion_entrenamiento=0.7, semilla=42):
    """Elige parámetros con una parte de las entidades y los valida en el resto.

    Elegir parámetros y medirlos con las mismas filas infla las métricas: la
    selección usa solo la parte de entrenamiento y el resultado que se reporta
    es el de la parte de prueba. El criterio es la exactitud balanceada, porque
    el F1 premia marcar todo cuando casi todo es positivo.
    """
    mask_tr, mask_te, n_ent_neg = _dividir_entidades(datos, proporcion_entrenamiento, semilla)
    if mask_tr.sum() == 0 or mask_te.sum() == 0:
        raise ValueError("No hay suficientes entidades etiquetadas para separar entrenamiento y prueba.")
    if n_ent_neg < 10:
        print(
            f"AVISO: solo hay {n_ent_neg} entidades con casos conformes. La validación por entidades "
            f"es poco confiable: trate el resultado de prueba como orientativo."
        )
    y = pd.to_numeric(datos["es_fraccionamiento"], errors="coerce").to_numpy()

    def bal(mask, puntaje, umbral):
        return balanced_accuracy_score(y[mask].astype(int), (puntaje[mask] >= umbral).astype(int))

    filas, cache = [], {}
    for ventana in VENTANAS:
        for unlp in UMBRALES_NLP:
            cache[(ventana, unlp)] = puntuar(datos, ventana, uit, unlp, contaminacion)["puntaje_riesgo"].to_numpy(float)
            for umbral in UMBRALES_PUNTAJE:
                filas.append({
                    "ventana": ventana, "umbral_nlp": unlp, "umbral_puntaje": umbral,
                    "exact_balanceada_entrenamiento": bal(mask_tr, cache[(ventana, unlp)], umbral),
                })
    tabla = pd.DataFrame(filas).sort_values(
        ["exact_balanceada_entrenamiento", "umbral_puntaje"], ascending=[False, False]
    ).reset_index(drop=True)
    mejor = tabla.iloc[0]
    clave = (int(mejor["ventana"]), float(mejor["umbral_nlp"]))
    umbral = float(mejor["umbral_puntaje"])
    base = puntuar(datos, 30, uit, 0.50, contaminacion)["puntaje_riesgo"].to_numpy(float)

    print(f"Filas etiquetadas: {int(mask_tr.sum())} de entrenamiento y {int(mask_te.sum())} de prueba.")
    print("\nMejores combinaciones según el conjunto de entrenamiento:")
    print(tabla.head(8).round(3).to_string(index=False))
    print("\nResultado en el conjunto de PRUEBA (no usado para elegir):")
    for nombre, puntaje, u in (
        ("Parámetros por defecto (30 días, 0.50, 71)", base, 71.0),
        (f"Calibrado ({clave[0]} días, {clave[1]:.2f}, {umbral:g})", cache[clave], umbral),
    ):
        pred = (puntaje[mask_te] >= u).astype(int)
        yt = y[mask_te].astype(int)
        tp, fn = int(((pred == 1) & (yt == 1)).sum()), int(((pred == 0) & (yt == 1)).sum())
        tn, fp = int(((pred == 0) & (yt == 0)).sum()), int(((pred == 1) & (yt == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else float("nan")
        espe = tn / (tn + fp) if tn + fp else float("nan")
        print(f"  {nombre}: sensibilidad {sens:.3f} | especificidad {espe:.3f} | exactitud balanceada {(sens + espe) / 2:.3f}")
    return tabla


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivo", type=Path, help="Excel o CSV con la etiqueta de fraccionamiento")
    parser.add_argument("--contexto", type=Path, nargs="+", help="Archivos con más compras (sin etiqueta) de las mismas entidades")
    parser.add_argument("--incluir-resumenes", action="store_true", help="No excluir las filas que son resúmenes de hallazgo")
    parser.add_argument("--ventana", type=int, default=30, help="Ventana de recurrencia en días (30)")
    parser.add_argument("--uit", type=float, default=5150.0, help="UIT para años sin valor oficial (5150)")
    parser.add_argument("--umbral-nlp", type=float, default=0.50, help="Similitud mínima entre descripciones (0.50)")
    parser.add_argument("--contaminacion", type=float, default=0.03, help="Proporción atípica esperada (0.03)")
    parser.add_argument("--umbral", type=float, default=UMBRAL_ALTO, help=f"Puntaje desde el cual se marca fraccionamiento ({UMBRAL_ALTO:g})")
    parser.add_argument("--calibrar", action="store_true", help="Busca ventana y umbrales con 70%% de las entidades y los valida con el 30%% restante")
    parser.add_argument("--salida", type=Path, help="Excel donde guardar resultados, métricas y errores")
    args = parser.parse_args(argv)

    if not args.archivo.exists():
        print(f"No existe el archivo: {args.archivo}", file=sys.stderr)
        return 1

    datos = normalizar_y_emparejar_columnas(_leer(args.archivo))
    omitidas = datos.attrs.get("filas_omitidas_fecha", 0)
    if omitidas:
        print(f"Aviso: {omitidas} filas sin fecha válida fueron omitidas.")
    if "es_fraccionamiento" not in datos.columns:
        print("El archivo no tiene una columna de etiqueta (es_fraccionamiento / label).", file=sys.stderr)
        return 1

    auditar_etiquetas(datos)
    if not args.incluir_resumenes:
        resumen = _es_resumen(datos)
        if resumen.any():
            print(f"\nSe excluyen {int(resumen.sum())} filas resumen de hallazgo (use --incluir-resumenes para conservarlas).")
            datos = datos.loc[~resumen].reset_index(drop=True)
    if args.contexto:
        datos = agregar_contexto(datos, args.contexto)

    if args.calibrar:
        print()
        calibrar(datos, args.uit, args.contaminacion)
        return 0

    resultados = puntuar(datos, args.ventana, args.uit, args.umbral_nlp, args.contaminacion)
    informe = evaluar_contra_etiquetas(resultados, umbral=args.umbral)
    print(
        f"\nEvaluadas {informe['total_evaluados']:,} filas "
        f"({informe['positivos']:,} con fraccionamiento, {informe['negativos']:,} conformes). "
        f"Umbral: {args.umbral:g} puntos."
    )
    _mostrar("Marcar todo como fraccionamiento (piso de referencia)", informe["referencia_todo_positivo"])
    _mostrar("Modelo híbrido (reglas + ajuste estadístico)", informe["modelo_hibrido"])
    if "solo_reglas" in informe:
        _mostrar("Solo reglas normativas", informe["solo_reglas"])

    barrido = barrido_de_umbrales(resultados)
    mejor = barrido.sort_values(["exactitud_balanceada", "umbral"], ascending=[False, False]).iloc[0]
    print(
        f"\nMejor exactitud balanceada en el barrido: {mejor['exactitud_balanceada']:.3f} con umbral {int(mejor['umbral'])} "
        f"(calculado sobre estos mismos datos; use --calibrar para una cifra validada)."
    )

    if args.salida:
        etiqueta = pd.to_numeric(resultados["es_fraccionamiento"], errors="coerce")
        pred = (resultados["puntaje_riesgo"] >= args.umbral).astype(int)
        errores = resultados[etiqueta.isin([0, 1]) & (pred != etiqueta)].copy()
        errores["tipo_error"] = [
            "Falso positivo" if p == 1 else "Falso negativo" for p in pred.loc[errores.index]
        ]
        columnas = [
            c for c in (
                "entidad_ruc", "entidad_nombre", "descripcion_item", "monto_referencial",
                "fecha_convocatoria", "codigo_cubso", "tipo_proceso", "es_fraccionamiento",
                "puntaje_riesgo", "score_reglas_negocio", "bono_ia", "nivel_riesgo",
                "frecuencia_compras_30d", "monto_acumulado_30d", "monto_acumulado_anual",
                "explicacion_reglas",
            ) if c in resultados.columns
        ]
        resumen_xlsx = pd.DataFrame([
            {"modelo": nombre, **informe[clave]}
            for nombre, clave in (
                ("Todo positivo", "referencia_todo_positivo"),
                ("Híbrido", "modelo_hibrido"),
                ("Solo reglas", "solo_reglas"),
            ) if clave in informe
        ])
        with pd.ExcelWriter(args.salida) as escritor:
            resultados[columnas].to_excel(escritor, sheet_name="Resultados", index=False)
            resumen_xlsx.to_excel(escritor, sheet_name="Metricas", index=False)
            barrido.to_excel(escritor, sheet_name="Umbrales", index=False)
            errores[columnas + ["tipo_error"]].to_excel(escritor, sheet_name="Errores", index=False)
        print(f"Resultados guardados en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
