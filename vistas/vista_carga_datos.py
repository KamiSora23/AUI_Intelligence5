from __future__ import annotations

import os
import unicodedata
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from base_datos.conexion_bd import obtener_engine, obtener_session
from base_datos.gestor_logs import registrar_log
from modulos_ia.extractor_caracteristicas import (
    es_resumen_de_hallazgo,
    extraer_caracteristicas_temporales_y_nlp,
)
from modulos_ia.aprendizaje_supervisado import aplicar as aplicar_modelo_supervisado
from modulos_ia.modelo_aislamiento import (
    UMBRAL_MEDIO,
    ejecutar_isolation_forest_solo,
    entrenar_y_evaluar_isolation_forest,
)
from utilidades.enviador_notificaciones import enviar_alerta_riesgo_critico
from utilidades import gestor_modelo_referencia
from utilidades.gestor_modelo_supervisado import modelo_activo
from utilidades.gestor_permisos import tiene_permiso
from utilidades.mapeador_columnas import (
    limpiar_cadena_codigo,
    limpiar_ruc,
    normalizar_fecha,
    normalizar_y_emparejar_columnas,
)
from vistas.componentes_ui import (
    encabezado_pagina,
    encabezado_seccion,
)


OPCIONES_INGESTA = ("Archivo local", "Conector OSCE", "Administrar lotes")


def leer_csv_con_codificacion(archivo) -> pd.DataFrame:
    """Lee un CSV en UTF-8 (con o sin BOM) o, si no lo es, en Windows-1252.

    Excel en Windows exporta CSV en Windows-1252; leerlo como UTF-8 falla o
    deja caracteres sustituidos en las descripciones.
    """
    try:
        return pd.read_csv(archivo, encoding="utf-8-sig")
    except UnicodeDecodeError:
        archivo.seek(0)
        return pd.read_csv(archivo, encoding="cp1252")


def evaluar_estado_segun_flujo(
    puntaje_riesgo: float,
    umbral_alerta_pts: float,
    modo_flujo: str,
) -> str:
    """Asigna el estado inicial según el flujo de revisión vigente.

    El puntaje es una señal de riesgo, no una confirmación jurídica: ni siquiera
    el flujo automático fija 'Confirmado_Fraccionamiento' o 'Descartado'; esos
    estados los asigna el auditor al dictaminar.

    El riesgo medio queda pendiente, sin propuesta, igual que en
    ``procesar_estado_revision``: son casos que activaron reglas de
    fraccionamiento y proponer descartarlos ocultaría fraccionamientos reales.
    """
    if modo_flujo not in ("Automático", "Semiautomático"):
        return "Pendiente"
    if puntaje_riesgo >= umbral_alerta_pts:
        return "Propuesta_Confirmado"
    if puntaje_riesgo >= UMBRAL_MEDIO:
        return "Pendiente"
    return "Propuesta_Descartado"


def renderizar_vista_carga() -> None:
    """Gestiona la incorporación, evaluación y persistencia de nuevas fuentes."""
    encabezado_pagina(
        "Ingesta de datos",
        "Incorpora archivos de contratación, valida su estructura y ejecuta el análisis de riesgos.",
        "upload",
        "Gestión de fuentes",
    )

    if not tiene_permiso("cargar_datos"):
        st.warning("Tu rol no permite incorporar ni eliminar fuentes de datos.")
        return

    # El conector OSCE se muestra solo si se activa en el .env.
    opciones = [
        opcion for opcion in OPCIONES_INGESTA
        if opcion != "Conector OSCE" or os.getenv("MOSTRAR_CONECTOR_OSCE", "no").strip().lower()
        in ("si", "sí", "true", "1")
    ]
    if st.session_state.get("modo_ingesta") not in (*opciones, None):
        st.session_state.pop("modo_ingesta")
    modo = st.segmented_control(
        "Modo de ingesta",
        opciones,
        default=opciones[0],
        key="modo_ingesta",
        label_visibility="collapsed",
        width="stretch",
    )

    fuente: tuple[pd.DataFrame | None, str] = (None, "")
    if modo == "Archivo local":
        fuente = _renderizar_archivo_local()
    elif modo == "Conector OSCE":
        fuente = _renderizar_conector_osce()
    else:
        _renderizar_gestion_lotes()

    datos, nombre_fuente = fuente
    if datos is not None and modo != "Administrar lotes":
        _renderizar_procesamiento(datos, nombre_fuente)


def _renderizar_archivo_local() -> tuple[pd.DataFrame | None, str]:
    encabezado_seccion(
        "Cargar archivo",
        "Formatos admitidos: Excel y CSV de hasta 200 MB.",
        "upload",
    )
    archivo = st.file_uploader(
        "Seleccione o arrastre el conjunto de datos de convocatorias (.xlsx, .csv)",
        type=["xlsx", "csv"],
        key="uploader_excel",
    )
    if archivo is None:
        return None, ""

    try:
        datos = (
            leer_csv_con_codificacion(archivo)
            if archivo.name.lower().endswith(".csv")
            else pd.read_excel(archivo)
        )
        st.info(
            f"Archivo validado: {len(datos):,} filas y {len(datos.columns):,} columnas."
        )
        return datos, archivo.name
    except Exception as error:
        st.error(f"No fue posible abrir el archivo: {error}")
        return None, ""


def _renderizar_conector_osce() -> tuple[pd.DataFrame | None, str]:
    encabezado_seccion(
        "Conector de datos OSCE",
        "Prepara una fuente local compatible cuando el servicio externo no está disponible.",
        "database",
    )
    st.caption(
        "La integración utiliza una fuente local de validación mientras los servicios "
        "externos del OSCE no estén disponibles."
    )

    columna_anio, columna_ruc = st.columns(2)
    with columna_anio:
        anio = st.selectbox("Año fiscal", [2026, 2025, 2024], index=0)
    with columna_ruc:
        ruc = st.text_input(
            "RUC de entidad (opcional)",
            placeholder="Ej. 20131378901",
        ).strip()

    if st.button("Preparar fuente local", type="primary"):
        datos = _crear_fuente_local()
        if ruc:
            datos = datos[datos["entidad_ruc"] == ruc].copy()

        if datos.empty:
            st.session_state.pop("fuente_osce_datos", None)
            st.session_state.pop("fuente_osce_nombre", None)
            st.warning("No se encontraron registros para el RUC indicado.")
        else:
            st.session_state["fuente_osce_datos"] = datos
            st.session_state["fuente_osce_nombre"] = (
                f"FUENTE_LOCAL_OSCE_{anio}.csv"
            )
            st.info(
                f"Fuente preparada correctamente: {len(datos):,} registros disponibles."
            )

    datos_preparados = st.session_state.get("fuente_osce_datos")
    nombre = st.session_state.get("fuente_osce_nombre", "")
    return datos_preparados, nombre


def _crear_fuente_local() -> pd.DataFrame:
    """Devuelve un conjunto estable para verificar el flujo de procesamiento."""
    return pd.DataFrame(
        [
            {
                "entidad_ruc": "20131378901",
                "entidad_nombre": "MUNICIPALIDAD PROVINCIAL DEL CALLAO",
                "descripcion_item": "ADQUISICIÓN DE PAPEL BOND A4 DE 75G PARA OFICINAS ADMINISTRATIVAS",
                "monto_referencial": 38500.00,
                "fecha_convocatoria": "2026-03-01",
                "codigo_cubso": "44121506",
            },
            {
                "entidad_ruc": "20131378901",
                "entidad_nombre": "MUNICIPALIDAD PROVINCIAL DEL CALLAO",
                "descripcion_item": "ADQUISICIÓN DE PAPEL BOND A4 75 GR PARA GERENCIA DE LOGÍSTICA",
                "monto_referencial": 39200.00,
                "fecha_convocatoria": "2026-03-05",
                "codigo_cubso": "44121506",
            },
            {
                "entidad_ruc": "20131378901",
                "entidad_nombre": "MUNICIPALIDAD PROVINCIAL DEL CALLAO",
                "descripcion_item": "COMPRA DE PAPEL BOND BLANCO TAMAÑO A4 PARA SUBGERENCIA",
                "monto_referencial": 37900.00,
                "fecha_convocatoria": "2026-03-12",
                "codigo_cubso": "44121506",
            },
            {
                "entidad_ruc": "20131378901",
                "entidad_nombre": "MUNICIPALIDAD PROVINCIAL DEL CALLAO",
                "descripcion_item": "SERVICIO DE MANTENIMIENTO PREVENTIVO DE COMPUTADORAS DE ESCRITORIO",
                "monto_referencial": 41000.00,
                "fecha_convocatoria": "2026-03-15",
                "codigo_cubso": "81111802",
            },
            {
                "entidad_ruc": "20504789123",
                "entidad_nombre": "HOSPITAL NACIONAL DOS DE MAYO",
                "descripcion_item": "ADQUISICIÓN DE MASCARILLAS QUIRÚRGICAS TRIPLE CAPA",
                "monto_referencial": 25000.00,
                "fecha_convocatoria": "2026-02-10",
                "codigo_cubso": "42131606",
            },
            {
                "entidad_ruc": "20504789123",
                "entidad_nombre": "HOSPITAL NACIONAL DOS DE MAYO",
                "descripcion_item": "ADQUISICIÓN DE MASCARILLAS QUIRÚRGICAS DE PROTECCIÓN",
                "monto_referencial": 24800.00,
                "fecha_convocatoria": "2026-02-14",
                "codigo_cubso": "42131606",
            },
            {
                "entidad_ruc": "20504789123",
                "entidad_nombre": "HOSPITAL NACIONAL DOS DE MAYO",
                "descripcion_item": "MASCARILLA DESECHABLE TRIPLE CAPA CON EMBALAJE HERMÉTICO",
                "monto_referencial": 26200.00,
                "fecha_convocatoria": "2026-02-20",
                "codigo_cubso": "42131606",
            },
            {
                "entidad_ruc": "20100012345",
                "entidad_nombre": "MINISTERIO DE EDUCACIÓN",
                "descripcion_item": "CONTRATACIÓN DE SERVICIO DE LIMPIEZA INTEGRAL DE SEDE CENTRAL",
                "monto_referencial": 180000.00,
                "fecha_convocatoria": "2026-01-15",
                "codigo_cubso": "76111501",
            },
        ]
    )


def _renderizar_gestion_lotes() -> None:
    encabezado_seccion(
        "Administrar lotes",
        "Consulta y retira cargas que ya no deban formar parte del análisis.",
        "database",
    )
    st.caption(
        "La eliminación afecta únicamente al lote seleccionado y a sus resultados asociados."
    )

    session = obtener_session()
    try:
        filas = session.execute(
            text(
                """
                SELECT id_carga, nombre_archivo, total_registros, fecha_carga
                FROM historial_cargas
                ORDER BY id_carga DESC
                """
            )
        ).fetchall()
        lotes = pd.DataFrame([dict(fila._mapping) for fila in filas])
    except Exception as error:
        st.error(f"No fue posible consultar el historial de cargas: {error}")
        return
    finally:
        session.close()

    if lotes.empty:
        st.info("No hay lotes cargados actualmente.")
        return

    registros = lotes.set_index("id_carga").to_dict("index")
    id_lote = st.selectbox(
        "Lote que desea administrar",
        options=list(registros),
        format_func=lambda identificador: _etiqueta_lote(
            identificador,
            registros[identificador],
        ),
    )
    st.warning(
        f"Esta acción eliminará el lote {id_lote} y todos sus resultados asociados."
    )
    if st.button("Eliminar lote seleccionado", key="btn_eliminar_lote"):
        _eliminar_lote(int(id_lote))


def _etiqueta_lote(identificador: int, lote: dict[str, Any]) -> str:
    fecha = str(lote["fecha_carga"])[:16]
    return (
        f"Lote {identificador} · {lote['nombre_archivo']} · "
        f"{fecha} · {int(lote['total_registros']):,} registros"
    )


def _eliminar_lote(id_lote: int) -> None:
    session = obtener_session()
    try:
        ids = [
            int(fila[0])
            for fila in session.execute(
                text(
                    "SELECT id_convocatoria FROM convocatorias_procesadas "
                    "WHERE id_carga = :id"
                ),
                {"id": id_lote},
            ).fetchall()
        ]
        if ids:
            lista_ids = ",".join(str(identificador) for identificador in ids)
            session.execute(
                text(
                    "DELETE FROM resultados_anomalias "
                    f"WHERE id_convocatoria IN ({lista_ids})"
                )
            )
            session.execute(
                text(
                    "DELETE FROM convocatorias_procesadas WHERE id_carga = :id"
                ),
                {"id": id_lote},
            )
        session.execute(
            text("DELETE FROM historial_cargas WHERE id_carga = :id"),
            {"id": id_lote},
        )
        session.commit()
        st.cache_data.clear()
        registrar_log(
            st.session_state.get("id_usuario", 1),
            "Eliminación de lote",
            "Ingesta",
            f"Se eliminó el lote {id_lote} y sus convocatorias asociadas.",
        )
        st.success(f"El lote {id_lote} fue eliminado.")
        st.rerun()
    except Exception as error:
        session.rollback()
        st.error(f"No fue posible eliminar el lote: {error}")
    finally:
        session.close()


MODELO_B = "Modelo B · Sistema completo"
MODELO_A = "Modelo A · Isolation Forest solo"


def _renderizar_procesamiento(datos: pd.DataFrame, nombre_fuente: str) -> None:
    encabezado_seccion(
        "Procesamiento y análisis",
        "Normalización, extracción de variables, evaluación y persistencia.",
        "brain",
    )

    st.radio(
        "Modelo de detección",
        (MODELO_B, MODELO_A),
        key="modelo_ingesta",
        horizontal=True,
        help=(
            "Modelo B: reglas normativas, similitud semántica, recurrencia y ajuste "
            "estadístico. Modelo A: solo Isolation Forest con variables estructurales y "
            "estadísticas (monto, compras y gasto de la entidad), sin reglas, similitud "
            "ni recurrencia. Sirve como línea base para comparar ambos sobre el mismo archivo."
        ),
    )
    if st.session_state.get("modelo_ingesta") == MODELO_A:
        _renderizar_referencia_modelo_a()
    st.checkbox(
        "Excluir filas que son resúmenes de hallazgo (no son compras)",
        value=True,
        key="excluir_resumenes",
        help=(
            "Filas cuyo texto ya dice 'fraccionamiento' u otro hallazgo Y cuyo CUBSO es "
            "'no definido'. No describen una compra, por lo que no pueden evaluarse y "
            "distorsionan las alertas."
        ),
    )
    if st.button("Iniciar procesamiento", type="primary"):
        _procesar_fuente(datos, nombre_fuente)

    encabezado_seccion(
        "Vista previa",
        "Primeros registros de la fuente seleccionada.",
        "file",
    )
    vista_previa = datos.head(10)
    st.dataframe(
        vista_previa,
        use_container_width=True,
        hide_index=True,
    )


def _renderizar_referencia_modelo_a() -> None:
    """Estado y entrenamiento del modelo A con compras de referencia (CONOSCE)."""
    guardado = gestor_modelo_referencia.modelo_referencia()
    if guardado is None:
        st.info(
            "El modelo A no tiene compras de referencia: se entrenará con el propio archivo. "
            "Entrénalo con compras del CONOSCE para que aprenda cómo luce una compra habitual."
        )
    else:
        st.success(
            f"El modelo A usa un Isolation Forest entrenado con {guardado['compras']:,} compras "
            f"de {guardado['entidades']:,} entidades ({', '.join(guardado['archivos'])}; "
            f"contaminación {guardado['contaminacion']}, {guardado['fecha']})."
        )
    if not tiene_permiso("cambiar_configuracion"):
        return
    with st.expander("Entrenar el modelo A con compras de referencia"):
        st.caption(
            "Archivos sin etiqueta con muchas compras reales (p. ej., adjudicaciones del "
            "CONOSCE). Reemplazan al modelo guardado; no se guardan como lotes."
        )
        archivos = st.file_uploader(
            "Archivos de referencia (.xlsx, .csv)",
            type=["xlsx", "csv"],
            accept_multiple_files=True,
            key="uploader_referencia",
        )
        columnas = st.columns(2)
        if columnas[0].button("Entrenar y guardar", disabled=not archivos):
            try:
                tablas = [
                    leer_csv_con_codificacion(a) if a.name.lower().endswith(".csv") else pd.read_excel(a)
                    for a in archivos
                ]
                contaminacion = _obtener_parametros_pipeline()["frecuencia_contaminacion"]
                nuevo = gestor_modelo_referencia.entrenar_y_guardar(
                    tablas, [a.name for a in archivos], contaminacion)
                registrar_log(
                    st.session_state.get("id_usuario", 1),
                    "Modelo A entrenado con referencia",
                    "Ingesta",
                    f"{nuevo['compras']} compras de {', '.join(nuevo['archivos'])}.",
                )
                st.rerun()
            except Exception as error:
                st.error(f"No fue posible entrenar el modelo A: {error}")
        if guardado is not None and columnas[1].button("Quitar la referencia"):
            gestor_modelo_referencia.eliminar()
            st.rerun()


def _procesar_fuente(datos: pd.DataFrame, nombre_fuente: str) -> None:
    progreso = st.progress(0)
    estado = st.empty()
    parametros = _obtener_parametros_pipeline()

    try:
        estado.text("1/4 · Normalizando y validando columnas")
        normalizados = normalizar_y_emparejar_columnas(datos)
        vacias = int(normalizados.attrs.get("filas_vacias", 0))
        if vacias:
            st.info(
                f"Se ignoraron {vacias:,} filas vacías del archivo (sin entidad, "
                "descripción, monto ni fecha); no son compras."
            )
        omitidas = int(normalizados.attrs.get("filas_omitidas_fecha", 0))
        if omitidas:
            st.warning(
                f"Se omitieron {omitidas:,} filas sin fecha de convocatoria válida; "
                "sin fecha no es posible evaluar recurrencia."
            )
        if normalizados.empty:
            raise ValueError("Ninguna fila tiene una fecha de convocatoria válida.")
        if st.session_state.get("excluir_resumenes", True):
            resumen = es_resumen_de_hallazgo(normalizados)
            if resumen.any():
                st.info(
                    f"Se excluyeron {int(resumen.sum()):,} filas que son resúmenes de "
                    "hallazgo y no compras; no se evalúan ni se guardan."
                )
                normalizados = normalizados.loc[~resumen].reset_index(drop=True)
            if normalizados.empty:
                raise ValueError("Todas las filas eran resúmenes de hallazgo; no hay compras que evaluar.")
        progreso.progress(25)

        if st.session_state.get("modelo_ingesta") == MODELO_A:
            estado.text("2/4 · Calculando variables estructurales y estadísticas")
            progreso.progress(50)
            estado.text("3/4 · Evaluando con Isolation Forest (modelo A)")
            guardado = gestor_modelo_referencia.modelo_referencia()
            if guardado is not None and guardado["contaminacion"] != parametros["frecuencia_contaminacion"]:
                st.warning(
                    f"El modelo A se entrenó con contaminación {guardado['contaminacion']} y la "
                    f"configuración actual es {parametros['frecuencia_contaminacion']}; se usa la "
                    "del entrenamiento. Vuelve a entrenarlo para aplicar la nueva."
                )
            resultados = ejecutar_isolation_forest_solo(
                normalizados,
                contaminacion=parametros["frecuencia_contaminacion"],
                modelo=None if guardado is None else guardado["modelo"],
            )
            # El nombre del lote indica el modelo para distinguirlo al comparar.
            nombre_lote = f"{nombre_fuente} · {MODELO_A}"
            if guardado is not None:
                nombre_lote += f" (referencia {guardado['compras']:,} compras)"
        else:
            estado.text("2/4 · Calculando recurrencia y similitud temática")
            caracteristicas = extraer_caracteristicas_temporales_y_nlp(
                normalizados,
                ventana_dias=parametros["ventana_dias"],
                uit_vigente=parametros["monto_uit_vigente"],
                umbral_similitud_nlp=parametros["umbral_similitud_nlp"],
            )
            progreso.progress(50)

            estado.text("3/4 · Evaluando patrones estadísticos")
            resultados, _, _ = entrenar_y_evaluar_isolation_forest(
                caracteristicas,
                contaminacion=parametros["frecuencia_contaminacion"],
            )
            # Solo puntúa si un administrador activó un modelo que superó la
            # validación; si no, los resultados quedan exactamente como estaban.
            activo = modelo_activo(obtener_engine())
            if activo is not None:
                resultados = aplicar_modelo_supervisado(resultados, activo[1])
                st.info(
                    f"Puntaje calculado con el modelo supervisado activo (id {activo[0]}); "
                    "cada motivo conserva el puntaje de reglas + ajuste como referencia."
                )
            nombre_lote = nombre_fuente
        progreso.progress(75)

        estado.text("4/4 · Guardando resultados y registrando trazabilidad")
        alertas = _persistir_resultados(
            resultados,
            nombre_lote,
            parametros,
        )
        progreso.progress(100)
        estado.text("Procesamiento finalizado")

        usuario = st.session_state.get("id_usuario", 1)
        registrar_log(
            usuario,
            "Ingesta completada",
            "Ingesta",
            f"Se procesaron {len(resultados)} registros desde {nombre_fuente}.",
        )
        _enviar_alertas(alertas)

        promedio = float(resultados["puntaje_riesgo"].mean())
        st.success(
            f"Se procesaron {len(resultados):,} convocatorias. "
            f"Puntaje promedio: {promedio:.2f}."
        )
    except Exception as error:
        st.error(f"El procesamiento no pudo completarse: {error}")


def _obtener_parametros_pipeline() -> dict[str, Any]:
    parametros: dict[str, Any] = {
        "frecuencia_contaminacion": 0.03,
        "umbral_similitud_nlp": 0.50,
        "ventana_dias": 30,
        "monto_uit_vigente": 5150.0,
        "notificar_riesgo_alto": True,
        "umbral_alerta_notificacion": 0.85,
        "flujo_revision": "Automático",
    }
    session = obtener_session()
    try:
        fila = session.execute(
            text(
                "SELECT * FROM configuracion_sistema "
                "ORDER BY id_config DESC LIMIT 1"
            )
        ).fetchone()
        if fila:
            parametros.update(dict(fila._mapping))
    finally:
        session.close()

    parametros["frecuencia_contaminacion"] = min(
        max(float(parametros["frecuencia_contaminacion"]), 0.001),
        0.50,
    )
    parametros["ventana_dias"] = max(int(parametros["ventana_dias"]), 1)
    parametros["monto_uit_vigente"] = max(
        float(parametros["monto_uit_vigente"]),
        1.0,
    )
    return parametros


def _persistir_resultados(
    resultados: pd.DataFrame,
    nombre_fuente: str,
    parametros: dict[str, Any],
) -> list[tuple[int, str, str, float, float]]:
    session = obtener_session()
    alertas: list[tuple[int, str, str, float, float]] = []
    try:
        carga = session.execute(
            text(
                """
                INSERT INTO historial_cargas
                    (nombre_archivo, total_registros, ventana_dias, id_usuario)
                VALUES (:nombre, :total, :ventana, :usuario)
                """
            ),
            {
                "nombre": nombre_fuente,
                "total": len(resultados),
                "ventana": int(parametros["ventana_dias"]),
                "usuario": st.session_state.get("id_usuario", 1),
            },
        )
        id_carga = carga.lastrowid or session.execute(
            text("SELECT MAX(id_carga) FROM historial_cargas")
        ).scalar()
        id_modelo = session.execute(
            text(
                "SELECT id_modelo FROM modelos_ia "
                "WHERE estado_activo = 1 LIMIT 1"
            )
        ).scalar() or 1

        # El umbral se guarda en escala 0-1 (p. ej. 0.85); el puntaje va de 0 a 100.
        umbral_alerta_pts = float(parametros.get("umbral_alerta_notificacion", 0.85))
        if umbral_alerta_pts <= 1.0:
            umbral_alerta_pts *= 100.0

        for _, fila in resultados.iterrows():
            convocatoria = session.execute(
                text(
                    """
                    INSERT INTO convocatorias_procesadas
                        (id_carga, entidad_ruc, entidad_nombre, descripcion_item,
                         monto_referencial, fecha_convocatoria, codigo_cubso,
                         tipo_proceso)
                    VALUES
                        (:carga, :ruc, :entidad, :descripcion, :monto, :fecha,
                         :cubso, :tipo)
                    """
                ),
                {
                    "carga": id_carga,
                    "ruc": limpiar_ruc(fila["entidad_ruc"]),
                    "entidad": str(fila["entidad_nombre"]),
                    "descripcion": str(fila["descripcion_item"]),
                    "monto": float(fila["monto_referencial"]),
                    "fecha": normalizar_fecha(fila["fecha_convocatoria"]),
                    "cubso": limpiar_cadena_codigo(
                        fila.get("codigo_cubso", "GENERAL")
                    ),
                    "tipo": str(
                        fila.get("tipo_proceso", "Igual o menor a 8 UIT")
                    ),
                },
            )
            id_convocatoria = convocatoria.lastrowid or session.execute(
                text("SELECT MAX(id_convocatoria) FROM convocatorias_procesadas")
            ).scalar()

            puntaje = min(
                max(float(fila.get("puntaje_riesgo", 0.0)), 0.0),
                100.0,
            )
            umbral_puntos = 71.0
            estado_revision = evaluar_estado_segun_flujo(
                puntaje,
                umbral_puntos,
                str(parametros.get("flujo_revision", "Automático")),
            )
            es_anomalia = int(
                puntaje >= 71.0
                or estado_revision in {"Confirmado_Fraccionamiento", "Propuesta_Confirmado"}
            )
            session.execute(
                text(
                    """
                    INSERT INTO resultados_anomalias
                        (id_convocatoria, id_modelo, similitud_nlp,
                         monto_acumulado_30d, frecuencia_compras_30d,
                         dias_desde_compra_anterior, score_reglas_negocio,
                         bono_ia, puntaje_riesgo, nivel_riesgo, es_anomalia,
                         explicacion_reglas, estado_revision, etiqueta_referencia)
                    VALUES
                        (:convocatoria, :modelo, :similitud, :acumulado,
                         :frecuencia, :dias, :reglas, :bono, :puntaje, :nivel,
                         :anomalia, :explicacion, :estado_revision, :etiqueta)
                    """
                ),
                {
                    "convocatoria": id_convocatoria,
                    "modelo": id_modelo,
                    "similitud": float(fila.get("similitud_nlp", 0.0)),
                    "acumulado": float(
                        fila.get("monto_acumulado_30d", 0.0)
                    ),
                    "frecuencia": int(
                        fila.get("frecuencia_compras_30d", 1)
                    ),
                    "dias": int(
                        fila.get("dias_desde_compra_anterior", 999)
                    ),
                    "reglas": float(
                        fila.get("score_reglas_negocio", 0.0)
                    ),
                    "bono": float(fila.get("bono_ia", 0.0)),
                    "puntaje": puntaje,
                    "nivel": _nivel_riesgo(puntaje),
                    "anomalia": es_anomalia,
                    "estado_revision": estado_revision,
                    "etiqueta": _etiqueta_referencia(fila),
                    "explicacion": str(
                        fila.get(
                            "explicacion_reglas",
                            "Evaluación automatizada",
                        )
                    ),
                },
            )

            if (
                bool(parametros.get("notificar_riesgo_alto", True))
                and puntaje >= umbral_alerta_pts
            ):
                alertas.append(
                    (
                        int(id_convocatoria),
                        str(fila["entidad_ruc"]),
                        str(fila["entidad_nombre"]),
                        puntaje,
                        float(fila["monto_referencial"]),
                    )
                )

        session.commit()
        st.cache_data.clear()
        return alertas
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


ETIQUETAS_TEXTO = {
    "FRACCIONAMIENTO": 1, "SI": 1, "VERDADERO": 1, "TRUE": 1,
    "NORMAL": 0, "CONFORME": 0, "NO": 0, "FALSO": 0, "FALSE": 0,
}


def _etiqueta_referencia(fila) -> int | None:
    """Etiqueta 0/1 de la fuente (p. ej. caso confirmado por la Contraloría).

    Acepta 0/1 o texto como el de ``caso_esperado`` ('FRACCIONAMIENTO' /
    'NORMAL'). Solo se guarda para medir el desempeño; no interviene en el
    puntaje.
    """
    valor = fila.get("es_fraccionamiento")
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        texto = unicodedata.normalize("NFKD", str(valor or "")).encode("ascii", "ignore")
        return ETIQUETAS_TEXTO.get(texto.decode().strip().upper())
    return int(numero) if numero in (0.0, 1.0) else None


def _nivel_riesgo(puntaje: float) -> str:
    if puntaje >= 71.0:
        return "Alto"
    if puntaje >= 41.0:
        return "Medio"
    return "Bajo"


def _enviar_alertas(
    alertas: list[tuple[int, str, str, float, float]],
) -> None:
    for alerta in alertas:
        try:
            enviar_alerta_riesgo_critico(*alerta)
        except Exception:
            # Una notificación no debe revertir resultados ya confirmados.
            continue
