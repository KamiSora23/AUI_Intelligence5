import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
from base_datos.conexion_bd import obtener_engine, obtener_session
from modulos_ia.modelo_aislamiento import orden_de_prioridad
from utilidades.estados_revision import describir_estado_revision, texto_ventana
from utilidades.exportador_resultados import construir_excel_resultados
from utilidades.gestor_lotes import renderizar_selector_lote_global, ventana_del_lote
from utilidades.paginacion import paginar_dataframe
from vistas.componentes_ui import (
    celda_apilada,
    encabezado_pagina,
    encabezado_seccion,
    estado_html,
    estilizar_grafica,
    resumen_tabla,
    tabla_responsiva,
    tarjeta_metrica,
    texto_seguro,
)


@st.cache_data(ttl=30, show_spinner=False)
def _cargar_datos_dashboard(lote_id: int) -> pd.DataFrame:
    """Recupera únicamente las operaciones del lote activo."""
    session = obtener_session()
    try:
        rows = session.execute(
            text(
                """
                SELECT
                    r.id_resultado,
                    c.id_convocatoria,
                    c.id_carga,
                    c.entidad_ruc,
                    c.entidad_nombre,
                    c.descripcion_item,
                    c.monto_referencial,
                    c.fecha_convocatoria,
                    c.codigo_cubso,
                    c.tipo_proceso,
                    r.similitud_nlp,
                    r.monto_acumulado_30d,
                    r.frecuencia_compras_30d,
                    r.dias_desde_compra_anterior,
                    r.score_reglas_negocio,
                    r.bono_ia,
                    r.atipicidad_ia,
                    r.puntaje_riesgo,
                    r.nivel_riesgo,
                    r.es_anomalia,
                    r.explicacion_reglas,
                    r.estado_revision,
                    r.id_usuario_auditor,
                    r.etiqueta_referencia,
                    r.comentario_auditor
                FROM resultados_anomalias r
                JOIN convocatorias_procesadas c
                    ON r.id_convocatoria = c.id_convocatoria
                WHERE c.id_carga = :lote
                -- Empates de puntaje: primero la compra más atípica según
                -- el Isolation Forest; si también empatan, la primera guardada.
                ORDER BY r.puntaje_riesgo DESC,
                         r.atipicidad_ia IS NULL, r.atipicidad_ia DESC,
                         r.id_resultado
                """
            ),
            {"lote": lote_id},
        ).fetchall()
        return (
            pd.DataFrame([dict(row._mapping) for row in rows])
            if rows
            else pd.DataFrame()
        )
    finally:
        session.close()


def ordenar_por_riesgo(datos: pd.DataFrame) -> pd.DataFrame:
    """Mayor puntaje primero; los empates, por atipicidad del Isolation Forest."""
    if datos.empty or "puntaje_riesgo" not in datos.columns:
        return datos.copy()
    atipicidad = (
        pd.to_numeric(datos["atipicidad_ia"], errors="coerce")
        if "atipicidad_ia" in datos.columns
        else pd.Series(float("nan"), index=datos.index)
    )
    orden = orden_de_prioridad(pd.to_numeric(datos["puntaje_riesgo"], errors="coerce").fillna(0.0), atipicidad)
    return datos.iloc[orden].copy()


def _preparar_paginas_dashboard(
    datos: pd.DataFrame,
    pagina_matriz: int,
    pagina_casos: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[int, int]]]:
    """Ordena y pagina las dos vistas extensas sin alterar los datos fuente."""
    ordenados = ordenar_por_riesgo(datos)
    matriz, actual_matriz, total_matriz = paginar_dataframe(
        ordenados,
        pagina_matriz,
    )
    casos, actual_casos, total_casos = paginar_dataframe(
        ordenados,
        pagina_casos,
    )
    return matriz, casos, {
        "matriz": (actual_matriz, total_matriz),
        "casos": (actual_casos, total_casos),
    }


@st.cache_data(ttl=30, show_spinner=False)
def _cargar_umbral_critico() -> float:
    """Puntaje de riesgo crítico: el mismo que dispara la notificación (escala 0-100)."""
    session = obtener_session()
    try:
        valor = session.execute(
            text(
                "SELECT umbral_alerta_notificacion FROM configuracion_sistema "
                "ORDER BY id_config DESC LIMIT 1"
            )
        ).scalar()
    finally:
        session.close()
    valor = float(valor) if valor is not None else 0.85
    return valor * 100.0 if valor <= 1.0 else valor


def _estado_revision_html(fila) -> str:
    """Etiqueta del estado de revisión con su aclaración debajo."""
    etiqueta, detalle = describir_estado_revision(
        fila["estado_revision"],
        pd.notna(fila.get("id_usuario_auditor")),
    )
    return (
        f"{estado_html(etiqueta)}"
        f"<span class='table-cell__meta'>{texto_seguro(detalle)}</span>"
    )


def _abrir_expediente(id_convocatoria: int) -> None:
    """Navega al expediente sin forzar una segunda ejecución de Streamlit."""
    st.session_state["caso_seleccionado_id"] = int(id_convocatoria)
    st.session_state["vista_actual"] = "vista_detalle_auditoria"


def renderizar_vista_dashboard():
    """Presenta los indicadores operativos y el detalle de casos procesados."""
    encabezado_pagina(
        "Centro de control",
        "Visión consolidada de riesgos, montos y patrones detectados en las contrataciones analizadas.",
        "dashboard",
        "Operación y monitoreo",
    )

    lote_id = renderizar_selector_lote_global(
        obtener_engine(),
        clave_componente="sel_dashboard_global",
    )
    if lote_id is None:
        return

    try:
        df_data = _cargar_datos_dashboard(lote_id)
        df_data = df_data.copy()

        df_data['monto_referencial'] = pd.to_numeric(df_data['monto_referencial'], errors='coerce').fillna(0.0)
        df_data['monto_acumulado_30d'] = pd.to_numeric(df_data['monto_acumulado_30d'], errors='coerce').fillna(0.0)
        df_data['frecuencia_compras_30d'] = pd.to_numeric(df_data['frecuencia_compras_30d'], errors='coerce').fillna(0).astype(int)
        df_data['dias_desde_compra_anterior'] = pd.to_numeric(df_data['dias_desde_compra_anterior'], errors='coerce').fillna(0).astype(int)
        df_data['puntaje_riesgo'] = pd.to_numeric(df_data['puntaje_riesgo'], errors='coerce').fillna(0.0)
        df_data['similitud_nlp'] = pd.to_numeric(df_data['similitud_nlp'], errors='coerce').fillna(0.0)
        df_data['max_similitud_nlp'] = (df_data['similitud_nlp'] * 100.0).round(1).astype(str) + "%"
        df_data['es_anomalia'] = pd.to_numeric(df_data['es_anomalia'], errors='coerce').fillna(0).astype(int)
        df_data['nivel_riesgo'] = df_data['nivel_riesgo'].fillna('Bajo')
        df_data['explicacion_reglas'] = df_data['explicacion_reglas'].fillna('Exento o sin indicios')
    except Exception as e:
        st.error(f"Error al cargar datos del dashboard: {e}")
        return

    if df_data.empty:
        st.info("No hay convocatorias registradas. Utiliza Ingesta de datos para cargar una fuente.")
        return

    ventana_dias = ventana_del_lote(lote_id)
    ventana = texto_ventana(ventana_dias)
    st.caption(f"Ámbito activo: resultados correspondientes al lote {lote_id}")
    if df_data.empty:
        st.info("El archivo seleccionado no registra convocatorias.")
        return

    # Resumen operativo.
    total_reg = len(df_data)
    anomalias_cnt = int((df_data['puntaje_riesgo'] >= 71.0).sum())
    # Prioridad dentro del riesgo alto; no cambia niveles ni propuestas.
    umbral_critico = _cargar_umbral_critico()
    criticos_cnt = int((df_data['puntaje_riesgo'] >= umbral_critico).sum())
    monto_sospechoso = df_data[df_data['puntaje_riesgo'] >= 71.0]['monto_referencial'].sum()
    risk_score_prom = df_data['puntaje_riesgo'].mean()

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        tarjeta_metrica(
            "Total de operaciones",
            f"{total_reg:,}",
            "Procesadas en el sistema",
            "file",
        )

    with col_k2:
        tarjeta_metrica(
            "Casos de riesgo alto",
            f"{anomalias_cnt:,}",
            f"71 puntos o más · {criticos_cnt:,} de riesgo crítico ({umbral_critico:.0f} o más)",
            "shield",
            "danger",
        )

    with col_k3:
        tarjeta_metrica(
            "Exposición observada",
            f"S/ {monto_sospechoso:,.2f}",
            "Monto de los casos de riesgo alto",
            "activity",
            "warning",
        )

    with col_k4:
        tarjeta_metrica(
            "Riesgo promedio",
            f"{risk_score_prom:.1f}",
            "Escala consolidada de 0 a 100",
            "chart",
            "success",
        )

    # Filtros operativos.
    with st.expander("Filtros del análisis", expanded=True):
        col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
        with col_f1:
            entidades_unicas = ["Todas"] + list(df_data['entidad_nombre'].unique())
            entidad_sel = st.selectbox("Entidad", entidades_unicas)
        with col_f2:
            opcion_critico = f"Crítico ({umbral_critico:.0f}–100)"
            nivel_riesgo_sel = st.selectbox(
                "Nivel de riesgo",
                ["Todos", opcion_critico, "Alto (71–100)", "Medio (41–70)", "Bajo (0–40)"],
            )
        with col_f3:
            estado_rev_sel = st.selectbox(
                "Estado de revisión",
                [
                    "Todos", "Pendiente", "Propuesta de confirmar", "Propuesta de descartar",
                    "Fraccionamiento confirmado", "Descartado",
                ],
            )

    # Aplicación de filtros.
    df_filtrado = df_data.copy()
    if entidad_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado['entidad_nombre'] == entidad_sel]
    if nivel_riesgo_sel == opcion_critico:
        df_filtrado = df_filtrado[df_filtrado['puntaje_riesgo'] >= umbral_critico]
    elif nivel_riesgo_sel == "Alto (71–100)":
        df_filtrado = df_filtrado[df_filtrado['puntaje_riesgo'] >= 71.0]
    elif nivel_riesgo_sel == "Medio (41–70)":
        df_filtrado = df_filtrado[(df_filtrado['puntaje_riesgo'] >= 41.0) & (df_filtrado['puntaje_riesgo'] < 71.0)]
    elif nivel_riesgo_sel == "Bajo (0–40)":
        df_filtrado = df_filtrado[df_filtrado['puntaje_riesgo'] < 41.0]

    if estado_rev_sel != "Todos":
        estado_db = {
            "Fraccionamiento confirmado": "Confirmado_Fraccionamiento",
            "Propuesta de confirmar": "Propuesta_Confirmado",
            "Propuesta de descartar": "Propuesta_Descartado",
        }.get(estado_rev_sel, estado_rev_sel)
        df_filtrado = df_filtrado[df_filtrado['estado_revision'] == estado_db]

    # Plotly requiere tipos numéricos nativos.
    df_filtrado['frecuencia_compras_30d'] = pd.to_numeric(df_filtrado['frecuencia_compras_30d'], errors='coerce').fillna(0).astype(int)
    df_filtrado['monto_acumulado_30d'] = pd.to_numeric(df_filtrado['monto_acumulado_30d'], errors='coerce').fillna(0.0).astype(float)
    df_filtrado['monto_referencial'] = pd.to_numeric(df_filtrado['monto_referencial'], errors='coerce').fillna(0.0).astype(float)
    df_filtrado['puntaje_riesgo'] = pd.to_numeric(df_filtrado['puntaje_riesgo'], errors='coerce').fillna(0.0).astype(float)

    # Distribuciones principales.
    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        with st.container(border=True):
            encabezado_seccion(
                "Frecuencia y monto acumulado",
                "Distribución de operaciones en la ventana temporal.",
                "chart",
            )
            montos_arr = df_filtrado['monto_referencial'].values
            m_min, m_max = (montos_arr.min(), montos_arr.max()) if len(montos_arr) > 0 else (0, 1)
            sizes_arr = 8 + (montos_arr - m_min) / (m_max - m_min + 1e-5) * 18

            hover_texts = [
                f"<b>{row['entidad_nombre']}</b><br>Proceso: {row['tipo_proceso']}<br>Monto: S/ {row['monto_referencial']:,.2f}<br>Acumulado en {ventana}: S/ {row['monto_acumulado_30d']:,.2f}<br>Puntaje: {row['puntaje_riesgo']:.1f}<br>Fundamento: {row['explicacion_reglas']}"
                for _, row in df_filtrado.iterrows()
            ]

            fig_scatter = go.Figure(data=go.Scatter(
                x=df_filtrado["frecuencia_compras_30d"].values,
                y=df_filtrado["monto_acumulado_30d"].values,
                mode="markers",
                marker=dict(
                    size=sizes_arr,
                    color=df_filtrado["puntaje_riesgo"].values,
                    colorscale=[
                        [0.0, "#4DD6A5"],
                        [0.55, "#F2B84B"],
                        [1.0, "#FF6F87"],
                    ],
                    showscale=True,
                    colorbar=dict(
                        title="Riesgo",
                        thickness=8,
                        outlinewidth=0,
                    ),
                    line=dict(width=.7, color="#D8D3FF"),
                ),
                text=hover_texts,
                hoverinfo="text",
            ))
            fig_scatter.update_layout(
                xaxis_title=f"Operaciones en {ventana}",
                yaxis_title="Monto acumulado (PEN)",
            )
            estilizar_grafica(fig_scatter, altura=330, mostrar_leyenda=False)
            st.plotly_chart(
                fig_scatter,
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )

    with col_g2:
        with st.container(border=True):
            encabezado_seccion(
                "Distribución de montos",
                "Comparación por nivel de riesgo.",
                "activity",
            )
            fig_box = px.box(
                df_filtrado,
                x="nivel_riesgo",
                y="monto_referencial",
                color="nivel_riesgo",
                points="all",
                labels={
                    "nivel_riesgo": "Nivel de riesgo",
                    "monto_referencial": "Monto referencial (PEN)",
                },
                color_discrete_map={
                    "Alto": "#FF6F87",
                    "Medio": "#F2B84B",
                    "Bajo": "#4DD6A5",
                },
            )
            estilizar_grafica(fig_box, altura=330, mostrar_leyenda=False)
            st.plotly_chart(
                fig_box,
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )

    df_matriz, df_casos, paginas = _preparar_paginas_dashboard(
        df_filtrado,
        int(st.session_state.get("pagina_matriz", 1)),
        int(st.session_state.get("pagina_casos", 1)),
    )
    pagina_matriz, total_paginas_matriz = paginas["matriz"]
    pagina_casos, total_paginas_casos = paginas["casos"]
    st.session_state["pagina_matriz"] = pagina_matriz
    st.session_state["pagina_casos"] = pagina_casos

    # Matriz de resultados para consulta y exportación.
    encabezado_seccion("Matriz de resultados", "Variables, reglas aplicadas y puntuación de cada operación.", "file")

    # Se exporta el lote completo, no solo lo filtrado: las métricas del libro
    # deben corresponder a todo el lote. El libro se genera al pulsar.
    datos_lote = df_data.copy()
    st.download_button(
        "Descargar resultados en Excel",
        data=lambda: construir_excel_resultados(datos_lote),
        file_name=f"resultados_lote_{lote_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
        help=(
            f"Las {len(datos_lote):,} operaciones del lote {lote_id} con las hojas "
            "Resultados, Metricas, Umbrales y Errores. Las tres últimas se llenan "
            "con la etiqueta del archivo (es_fraccionamiento o caso_esperado) o, "
            "si no la trae, con los dictámenes registrados por un auditor."
        ),
        on_click="ignore",
    )

    st.number_input(
        "Página de la matriz",
        min_value=1,
        max_value=total_paginas_matriz,
        step=1,
        key="pagina_matriz",
    )
    st.caption(
        f"Página {pagina_matriz} de {total_paginas_matriz} · "
        "máximo 50 operaciones por página."
    )

    df_render = df_matriz
    filas_resultados = []
    for _, fila in ordenar_por_riesgo(df_render).iterrows():
        puntaje = float(fila["puntaje_riesgo"])
        tono = "high" if puntaje >= 71 else ("medium" if puntaje >= 41 else "low")
        nivel = "Alto" if puntaje >= 71 else ("Medio" if puntaje >= 41 else "Bajo")
        riesgo = (
            f"{estado_html(nivel, tono)}"
            f"<span class='table-cell__meta'>{puntaje:.1f} puntos</span>"
            + (
                f"<span class='table-cell__meta'><b>Riesgo crítico</b></span>"
                if puntaje >= umbral_critico else ""
            )
        )
        filas_resultados.append(
            [
                riesgo,
                _estado_revision_html(fila),
                celda_apilada(fila["entidad_nombre"], fila["descripcion_item"]),
                celda_apilada(
                    fila["tipo_proceso"],
                    f"RUC {fila['entidad_ruc']}",
                ),
                celda_apilada(
                    f"S/ {float(fila['monto_referencial']):,.2f}",
                    (
                        f"Acumulado en {ventana}: "
                        f"S/ {float(fila['monto_acumulado_30d']):,.2f}"
                    ),
                ),
                celda_apilada(
                    fila["explicacion_reglas"],
                    f"{int(fila['frecuencia_compras_30d'])} operaciones en {ventana}",
                    f"Similitud temática: {fila['max_similitud_nlp']}",
                ),
            ]
        )

    with st.container(border=True, height=550):
        resumen_tabla(
            len(df_render),
            "Ordenada por prioridad, con el contenido completo distribuido en varias líneas.",
            "operaciones",
        )
        tabla_responsiva(
            ["Riesgo", "Revisión", "Contratación", "Proceso", "Montos", "Señales detectadas"],
            filas_resultados,
            ["10%", "13%", "25%", "13%", "15%", "24%"],
            "Matriz de resultados de riesgo",
        )

    encabezado_seccion("Casos para revisión", "Acceso directo al expediente de cada operación.", "filter")

    st.number_input(
        "Página de casos",
        min_value=1,
        max_value=total_paginas_casos,
        step=1,
        key="pagina_casos",
    )
    st.caption(
        f"Página {pagina_casos} de {total_paginas_casos} · "
        "máximo 50 expedientes por página."
    )

    with st.container(height=550, border=False):
        for _, row in df_casos.iterrows():
            with st.container(border=True):
                st.markdown(
                    '<span class="case-card-marker" aria-hidden="true"></span>',
                    unsafe_allow_html=True,
                )
                col_t1, col_t2, col_t3, col_t4 = st.columns(
                    [1.45, 3.8, 2.8, 1.25]
                )

                with col_t1:
                    tono = (
                        "high"
                        if row["puntaje_riesgo"] >= 71
                        else (
                            "medium"
                            if row["puntaje_riesgo"] >= 41
                            else "low"
                        )
                    )
                    nivel_mostrado = (
                        "Alto"
                        if row["puntaje_riesgo"] >= 71
                        else (
                            "Medio"
                            if row["puntaje_riesgo"] >= 41
                            else "Bajo"
                        )
                    )
                    st.markdown(
                        (
                            '<div class="case-card__risk">'
                            f"{estado_html(nivel_mostrado, tono)}"
                            f"<span>{float(row['puntaje_riesgo']):.1f} puntos</span>"
                            + (
                                "<small><b>Riesgo crítico</b></small>"
                                if row["puntaje_riesgo"] >= umbral_critico else ""
                            )
                            + f"<small>RUC {texto_seguro(row['entidad_ruc'])}</small>"
                            "</div>"
                            '<div class="case-card__risk">'
                            f"{_estado_revision_html(row)}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                with col_t2:
                    st.markdown(
                        celda_apilada(
                            row["entidad_nombre"],
                            row["descripcion_item"],
                            f"Proceso: {row['tipo_proceso']}",
                        ),
                        unsafe_allow_html=True,
                    )
                with col_t3:
                    st.markdown(
                        celda_apilada(
                            f"S/ {float(row['monto_acumulado_30d']):,.2f}",
                            row["explicacion_reglas"],
                            (
                                f"{int(row['frecuencia_compras_30d'])} "
                                f"operaciones en {ventana}"
                            ),
                        ),
                        unsafe_allow_html=True,
                    )
                with col_t4:
                    st.button(
                        "Abrir",
                        icon=":material/arrow_outward:",
                        key=f"btn_audit_dash_{row['id_resultado']}",
                        width="stretch",
                        type="primary",
                        on_click=_abrir_expediente,
                        args=(int(row["id_convocatoria"]),),
                    )
