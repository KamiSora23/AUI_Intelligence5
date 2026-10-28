import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

UMBRAL_ALTO = 71.0
UMBRAL_MEDIO = 41.0
BONO_MAXIMO = 20.0
# Anomalía a partir de la cual el ajuste estadístico alcanza el máximo.
ESCALA_ANOMALIA = 0.15


def procesar_estado_revision(
    df: pd.DataFrame,
    modo_automatico: bool = True,
) -> pd.DataFrame:
    """Asigna el estado inicial según el flujo de revisión configurado.

    Un puntaje alto es una señal de riesgo, no una confirmación jurídica: el
    modo automático solo PROPONE un resultado. 'Confirmado_Fraccionamiento' y
    'Descartado' los fija únicamente el dictamen de un auditor.
    """
    df_out = df.copy()
    if modo_automatico:
        condiciones = [
            (df_out['puntaje_riesgo'] >= UMBRAL_ALTO),
            (df_out['puntaje_riesgo'] < UMBRAL_MEDIO),
        ]
        estados = ['Propuesta_Confirmado', 'Propuesta_Descartado']
        df_out['estado_revision'] = np.select(condiciones, estados, default='Pendiente')
    else:
        if 'estado_revision' not in df_out.columns:
            df_out['estado_revision'] = 'Pendiente'
    return df_out


def _matriz_variables(df: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    """Variables independientes del puntaje reglado (evita circularidad)."""
    sub = df.loc[mask]
    if 'ratio_monto_8uit' in sub.columns:
        ratio = sub['ratio_monto_8uit'].astype(float)
    else:
        ratio = sub['monto_acumulado_30d'].astype(float) / (8.0 * 5150.0)
    dias = sub['dias_desde_compra_anterior'].astype(float).clip(upper=365.0)
    return np.column_stack([
        sub['frecuencia_compras_30d'].astype(float),
        sub['max_similitud_nlp'].astype(float),
        np.log1p(ratio.clip(lower=0.0)),
        dias,
    ])


def ejecutar_modelo_aislamiento_hibrido(
    df_procesado: pd.DataFrame,
    contaminacion: float = 0.03,
    modo_automatico: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Suma al puntaje reglado un ajuste (0 a 20) solo a operaciones atípicas."""
    df = df_procesado.copy()

    if 'es_excluido_normativa' not in df.columns:
        df['es_excluido_normativa'] = False
    if 'max_similitud_nlp' not in df.columns:
        df['max_similitud_nlp'] = df.get('similitud_nlp', 0.0)
    if 'score_reglas_negocio' not in df.columns:
        df['score_reglas_negocio'] = 0.0

    mask_eval = ~df['es_excluido_normativa'].astype(bool)
    df['bono_ia'] = 0.0

    if mask_eval.sum() >= 3:
        X = StandardScaler().fit_transform(_matriz_variables(df, mask_eval))
        iso = IsolationForest(
            n_estimators=100,
            contamination=contaminacion,
            random_state=random_state,
        ).fit(X)
        # decision_function < 0 marca a las operaciones atípicas; las normales
        # (>= 0) no reciben ajuste.
        atipicidad = np.clip(-iso.decision_function(X) / ESCALA_ANOMALIA, 0.0, 1.0)
        df.loc[mask_eval, 'bono_ia'] = np.round(atipicidad * BONO_MAXIMO, 2)

    df['puntaje_riesgo'] = np.clip(df['score_reglas_negocio'] + df['bono_ia'], 0.0, 100.0)

    conds = [
        (df['puntaje_riesgo'] >= UMBRAL_ALTO),
        (df['puntaje_riesgo'] >= UMBRAL_MEDIO) & (df['puntaje_riesgo'] < UMBRAL_ALTO),
        (df['puntaje_riesgo'] < UMBRAL_MEDIO),
    ]
    df['nivel_riesgo'] = np.select(conds, ['Alto', 'Medio', 'Bajo'], default='Bajo')
    df['es_anomalia'] = (df['puntaje_riesgo'] >= UMBRAL_ALTO).astype(int)

    return procesar_estado_revision(df, modo_automatico=modo_automatico)


# --- Modelo A: Isolation Forest por sí solo -----------------------------------
# Variables estructurales y estadísticas calculadas desde el archivo, sin el
# extractor del sistema: allí la frecuencia, los acumulados y los días entre
# compras agrupan las compras por similitud de texto o mismo CUBSO dentro de
# una ventana, es decir, ya contienen el análisis semántico y de recurrencia
# del sistema completo. Tampoco se usa el umbral de 8 UIT, base de las reglas.
VARIABLES_IF_SOLO = (
    "log_monto",               # estructural: tamaño de la compra
    "log_compras_entidad",     # estadística: compras de la entidad en el archivo
    "log_monto_total_entidad", # estadística: gasto total de la entidad en el archivo
    "z_monto_entidad",         # estadística: cuán atípico es el monto dentro de su entidad
)
EXPLICACION_IF_SOLO = (
    "Modelo A (Isolation Forest solo, {origen}): compra {estado} (atipicidad {atipicidad:+.3f}) "
    "según su monto, las compras y el gasto total de la entidad y el monto frente "
    "a su entidad. Sin reglas normativas, similitud semántica ni recurrencia."
)


def variables_isolation_forest_solo(df: pd.DataFrame) -> pd.DataFrame:
    """Variables del modelo A, sin texto, CUBSO, ventanas ni 8 UIT."""
    monto = pd.to_numeric(df["monto_referencial"], errors="coerce").fillna(0.0).clip(lower=0.0)
    por_entidad = monto.groupby(df["entidad_ruc"].astype(str))
    desviacion = por_entidad.transform("std").fillna(0.0)
    z = (monto - por_entidad.transform("mean")) / desviacion.where(desviacion > 0)
    return pd.DataFrame({
        "log_monto": np.log1p(monto),
        "log_compras_entidad": np.log1p(por_entidad.transform("size")),
        "log_monto_total_entidad": np.log1p(por_entidad.transform("sum")),
        "z_monto_entidad": z.fillna(0.0),
    })[list(VARIABLES_IF_SOLO)]


def _nuevo_bosque(contaminacion: float, random_state: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=100, contamination=contaminacion, random_state=random_state),
    )


def entrenar_isolation_forest_referencia(
    referencia: pd.DataFrame, contaminacion: float = 0.03, random_state: int = 42
) -> Pipeline:
    """Entrena el modelo A con compras de referencia (p. ej., CONOSCE), sin etiqueta.

    El bosque aprende cómo luce una compra habitual en muchas compras reales y
    luego puntúa archivos nuevos sin reentrenarse con ellos. Las compras con
    monto 0 no describen una adquisición y se descartan.
    """
    monto = pd.to_numeric(referencia["monto_referencial"], errors="coerce").fillna(0.0)
    referencia = referencia.loc[monto > 0]
    if len(referencia) < 10:
        raise ValueError("Se necesitan al menos 10 compras de referencia con monto mayor a 0.")
    modelo = _nuevo_bosque(contaminacion, random_state)
    return modelo.fit(variables_isolation_forest_solo(referencia).to_numpy(dtype=float))


def ajustar_isolation_forest_solo(
    df: pd.DataFrame,
    contaminacion: float = 0.03,
    random_state: int = 42,
    modelo: Pipeline | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve qué compras son atípicas y su atipicidad (> 0 = atípica).

    Sin ``modelo``, el bosque se entrena con el propio archivo. Con un modelo
    entrenado con compras de referencia solo puntúa; las variables de entidad
    se calculan igualmente con el archivo evaluado.
    """
    X = variables_isolation_forest_solo(df).to_numpy(dtype=float)
    if modelo is None:
        modelo = _nuevo_bosque(contaminacion, random_state).fit(X)
    return modelo.predict(X) == -1, -modelo.decision_function(X)


def ejecutar_isolation_forest_solo(
    df_convocatorias: pd.DataFrame,
    contaminacion: float = 0.03,
    modo_automatico: bool = True,
    random_state: int = 42,
    modelo: Pipeline | None = None,
) -> pd.DataFrame:
    """Modelo A: el puntaje sale solo del Isolation Forest.

    Con ``modelo`` (entrenado con compras de referencia) el archivo solo se
    puntúa; sin él, el bosque se entrena con el propio archivo.

    El bosque solo decide si una compra es atípica o no, así que no hay nivel
    Medio: una compra atípica queda en riesgo Alto (71 a 100) y el resto en
    Bajo (0 a 40), en escalas que crecen con la atipicidad. Las columnas
    propias del sistema completo (reglas, similitud, recurrencia) quedan en
    valores neutros: este modelo no las calcula.
    """
    df = df_convocatorias.copy().reset_index(drop=True)
    atipica, atipicidad = ajustar_isolation_forest_solo(df, contaminacion, random_state, modelo)
    origen = "entrenado con compras de referencia" if modelo is not None else "entrenado con este archivo"
    proporcion = atipicidad / ESCALA_ANOMALIA
    alto = UMBRAL_ALTO + np.clip(proporcion, 0.0, 1.0) * (100.0 - UMBRAL_ALTO)
    bajo = (UMBRAL_MEDIO - 0.01) * np.clip(1.0 + proporcion, 0.0, 1.0)
    puntaje = np.where(atipica, alto, bajo)
    df["puntaje_riesgo"] = np.round(np.clip(puntaje, 0.0, 100.0), 2)
    df["score_reglas_negocio"] = 0.0
    df["bono_ia"] = df["puntaje_riesgo"]
    df["similitud_nlp"] = 0.0
    df["max_similitud_nlp"] = 0.0
    df["frecuencia_compras_30d"] = 1
    df["monto_acumulado_30d"] = pd.to_numeric(df["monto_referencial"], errors="coerce").fillna(0.0)
    df["dias_desde_compra_anterior"] = 999
    df["es_excluido_normativa"] = False
    df["explicacion_reglas"] = [
        EXPLICACION_IF_SOLO.format(origen=origen, estado="atípica" if a else "no atípica", atipicidad=v)
        for a, v in zip(atipica, atipicidad)
    ]
    conds = [
        (df["puntaje_riesgo"] >= UMBRAL_ALTO),
        (df["puntaje_riesgo"] >= UMBRAL_MEDIO),
    ]
    df["nivel_riesgo"] = np.select(conds, ["Alto", "Medio"], default="Bajo")
    df["es_anomalia"] = atipica.astype(int)
    return procesar_estado_revision(df, modo_automatico=modo_automatico)


def entrenar_y_evaluar_isolation_forest(
    df_caracteristicas: pd.DataFrame,
    contaminacion: float = 0.03,
    random_state: int = 42,
    modo_automatico: bool = True,
):
    """Mantiene la firma utilizada por la capa de ingesta."""
    df_res = ejecutar_modelo_aislamiento_hibrido(
        df_caracteristicas,
        contaminacion=contaminacion,
        modo_automatico=modo_automatico,
        random_state=random_state,
    )
    return df_res, None, None
