from textwrap import dedent

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from base_datos.conexion_bd import obtener_engine, obtener_session
from base_datos.gestor_logs import registrar_log
from modulos_ia.explicabilidad_ia import calcular_explicabilidad_caso
from modulos_ia.analizador_redes import construir_grafo_bipartito_fraccionamiento
from utilidades.generador_reportes_pdf import generar_dossier_pdf_auditoria
from utilidades.estados_revision import (
    describir_estado_revision,
    explicar_estado_revision,
    nivel_de_puntaje,
    texto_separacion,
    texto_ventana,
)
from utilidades.gestor_lotes import renderizar_selector_lote_global, ventana_del_lote
from utilidades.gestor_permisos import tiene_permiso
from vistas.componentes_ui import (
    encabezado_pagina,
    encabezado_seccion,
    estilizar_grafica,
    tarjeta_metrica,
)


@st.cache_data(ttl=30, show_spinner=False)
def _cargar_expedientes_lote(lote_id: int) -> pd.DataFrame:
    """Recupera la lista ligera de expedientes de un lote."""
    session = obtener_session()
    try:
        filas = session.execute(
            text(
                """
                SELECT c.id_convocatoria, c.entidad_nombre, r.puntaje_riesgo,
                       r.estado_revision, r.id_usuario_auditor
                FROM convocatorias_procesadas c
                JOIN resultados_anomalias r
                  ON c.id_convocatoria = r.id_convocatoria
                WHERE c.id_carga = :lote
                ORDER BY r.puntaje_riesgo DESC,
                         r.atipicidad_ia IS NULL, r.atipicidad_ia DESC,
                         r.id_resultado
                """
            ),
            {"lote": lote_id},
        ).fetchall()
        return pd.DataFrame([dict(fila._mapping) for fila in filas])
    finally:
        session.close()


@st.cache_data(ttl=30, show_spinner=False)
def _cargar_caso(lote_id: int, id_convocatoria: int) -> dict | None:
    """Recupera el expediente seleccionado dentro de su lote."""
    session = obtener_session()
    try:
        fila = session.execute(
            text(
                """
                SELECT c.*, r.id_resultado, r.similitud_nlp,
                       r.monto_acumulado_30d, r.frecuencia_compras_30d,
                       r.dias_desde_compra_anterior, r.puntaje_riesgo,
                       r.es_anomalia, r.estado_revision, r.id_usuario_auditor,
                       r.comentario_auditor,
                       r.score_reglas_negocio, r.bono_ia, r.explicacion_reglas
                FROM convocatorias_procesadas c
                JOIN resultados_anomalias r
                  ON c.id_convocatoria = r.id_convocatoria
                WHERE c.id_convocatoria = :id AND c.id_carga = :lote
                """
            ),
            {"id": id_convocatoria, "lote": lote_id},
        ).fetchone()
        return dict(fila._mapping) if fila else None
    finally:
        session.close()


@st.cache_data(ttl=30, show_spinner=False)
def _cargar_relaciones(lote_id: int, ruc: str) -> pd.DataFrame:
    """Recupera relaciones de la entidad limitadas al lote activo."""
    session = obtener_session()
    try:
        filas = session.execute(
            text(
                """
                SELECT c.*, r.puntaje_riesgo
                FROM convocatorias_procesadas c
                JOIN resultados_anomalias r
                  ON c.id_convocatoria = r.id_convocatoria
                WHERE c.id_carga = :lote AND c.entidad_ruc = :ruc
                """
            ),
            {"lote": lote_id, "ruc": ruc},
        ).fetchall()
        return pd.DataFrame([dict(fila._mapping) for fila in filas])
    finally:
        session.close()


@st.cache_data(ttl=30, show_spinner=False)
def _construir_grafo_cacheado(
    relaciones: pd.DataFrame,
    ruc: str,
):
    """Reutiliza la disposición NetworkX mientras los datos no cambien."""
    return construir_grafo_bipartito_fraccionamiento(relaciones, ruc)


def renderizar_vista_detalle():
    """Compone el expediente completo de la operación seleccionada."""
    encabezado_pagina(
        "Expediente de auditoría",
        "Evidencia, variables de riesgo, relaciones y decisión documentada del auditor.",
        "file",
        "Análisis de caso",
    )

    lote_id = renderizar_selector_lote_global(
        obtener_engine(),
        clave_componente="sel_expediente_global",
    )
    if lote_id is None:
        return

    try:
        expedientes = _cargar_expedientes_lote(lote_id)
        if expedientes.empty:
            st.info("El archivo seleccionado no registra convocatorias.")
            return

        opciones = {
            int(fila.id_convocatoria): (
                f"{fila.id_convocatoria} · {fila.entidad_nombre} · "
                f"{float(fila.puntaje_riesgo):.1f} pts · "
                f"{describir_estado_revision(fila.estado_revision, pd.notna(fila.id_usuario_auditor))[0]}"
            )
            for fila in expedientes.itertuples(index=False)
        }
        id_previo = st.session_state.get("caso_seleccionado_id")
        ids = list(opciones)
        id_conv_sel = st.selectbox(
            "Expediente activo",
            options=ids,
            format_func=opciones.__getitem__,
            index=ids.index(id_previo) if id_previo in ids else 0,
            key="expediente_activo_select",
        )
        st.session_state["caso_seleccionado_id"] = int(id_conv_sel)

        caso_data = _cargar_caso(lote_id, int(id_conv_sel))
    except Exception as e:
        st.error(f"Error al cargar detalle del caso: {e}")
        return

    if not caso_data:
        st.info("Selecciona una operación desde el Centro de control para abrir su expediente.")
        return

    # Banner de Alerta Principal
    tipo_proc = str(caso_data.get('tipo_proceso', 'Igual o Menor a 8 UIT'))
    raw_score = caso_data.get('puntaje_riesgo') or caso_data.get('score') or caso_data.get('puntaje') or 0.0
    score_normalizado = min(float(raw_score), 100.0)
    score_pts = score_normalizado

    if "Licitación" in tipo_proc or "Concurso" in tipo_proc or "Subasta" in tipo_proc:
        banner_bg = "rgba(77, 214, 165, .08)"
        banner_border = "#4DD6A5"
        banner_txt = f"Proceso competitivo · {score_normalizado:.1f} puntos"
    elif score_normalizado >= 71.0:
        banner_bg = "rgba(255, 111, 135, .08)"
        banner_border = "#FF6F87"
        banner_txt = f"Riesgo alto · {score_normalizado:.1f} puntos"
    elif score_normalizado >= 41.0:
        banner_bg = "rgba(242, 184, 75, .08)"
        banner_border = "#F2B84B"
        banner_txt = f"Riesgo medio · {score_normalizado:.1f} puntos"
    else:
        banner_bg = "rgba(77, 214, 165, .08)"
        banner_border = "#4DD6A5"
        banner_txt = f"Riesgo bajo · {score_normalizado:.1f} puntos"

    st.markdown(dedent(f"""
    <div style="background-color: {banner_bg}; border: 1px solid {banner_border}; border-radius: 12px; padding: 11px 13px; margin-bottom: 10px; text-align: left;">
        <h2 style="margin: 0; color: #F7F8FC; font-size: 1rem;">{banner_txt}</h2>
        <p style="margin: 3px 0 0 0; color: #8D97AA; font-size: .68rem;">Operación {caso_data['id_convocatoria']} · <b>{caso_data['entidad_nombre']}</b> · RUC {caso_data['entidad_ruc']}</p>
    </div>
    """), unsafe_allow_html=True)

    ventana_dias = ventana_del_lote(lote_id)
    ventana = texto_ventana(ventana_dias)

    # Resumen del caso: separa el nivel de riesgo de la decisión provisional.
    estado_revision = caso_data.get('estado_revision')
    con_auditor = caso_data.get('id_usuario_auditor') is not None
    etiqueta_estado, detalle_estado = describir_estado_revision(estado_revision, con_auditor)
    nivel = nivel_de_puntaje(score_normalizado)
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        tarjeta_metrica("Estado de revisión", etiqueta_estado, detalle_estado, "shield")
    with col_r2:
        tarjeta_metrica("Puntaje", f"{score_normalizado:.1f}/100", "Escala consolidada", "chart")
    with col_r3:
        tarjeta_metrica(
            "Nivel de riesgo", nivel, "Alto ≥ 71 · Medio ≥ 41", "activity",
            {"Alto": "danger", "Medio": "warning", "Bajo": "success"}[nivel],
        )
    with col_r4:
        tarjeta_metrica(
            "Ventana analizada",
            f"{ventana_dias} días" if ventana_dias else "No disponible",
            "Aplicada al procesar el lote", "file",
        )
    st.caption(explicar_estado_revision(estado_revision, con_auditor))

    # Cuadro de Justificación Explícita de Auditoría (Arquitectura Híbrida)
    freq = int(caso_data['frecuencia_compras_30d'])
    nlp = float(caso_data['similitud_nlp'])
    monto_acum = float(caso_data['monto_acumulado_30d'])
    dias_texto = texto_separacion(caso_data.get('dias_desde_compra_anterior'))

    # Puntajes y motivos tal como los calculó el motor; no se reconstruyen
    # aquí con reglas propias que podrían diferir de las aplicadas.
    subtotal_reglas = round(float(caso_data.get('score_reglas_negocio') or 0.0), 1)
    pts_bonus_ia = round(float(caso_data.get('bono_ia') or 0.0), 1)
    pts_total = round(score_pts, 1)
    motivos_reglas = [
        motivo.strip()
        for motivo in str(caso_data.get('explicacion_reglas') or '').split(';')
        if motivo.strip()
    ]

    if "Licitación" in tipo_proc or "Concurso" in tipo_proc or "Subasta" in tipo_proc:
        justificacion = (
            f"<b>Evaluación normativa</b><br>"
            f"Clasificación: <b>proceso competitivo</b>. Modalidad: <b>{tipo_proc}</b>.<br>"
            f"La modalidad cuenta con un mecanismo de competencia pública y no activa las reglas de recurrencia."
        )
        box_color = "rgba(77, 214, 165, .06)"
        border_c = "#4DD6A5"
    elif score_pts >= 71.0:
        justificacion = (
            f"<b>Evaluación combinada · {pts_total:.1f} de 100 puntos</b><br>"
            f"Clasificación: <b>riesgo alto</b>. Modalidad: <b>{tipo_proc}</b>.<br>"
            f"La entidad <b>{caso_data['entidad_nombre']}</b> registró <b>{freq} convocatorias recurrentes</b> "
            f"de '<i>{caso_data['descripcion_item'][:60]}...</i>' en {ventana}, acumulando <b>S/ {monto_acum:,.2f}</b> "
            f"con una similitud temática del <b>{nlp:.1%}</b>."
        )
        box_color = "rgba(255, 111, 135, .06)"
        border_c = "#FF6F87"
    elif score_pts >= 41.0:
        justificacion = (
            f"<b>Evaluación combinada · {pts_total:.1f} de 100 puntos</b><br>"
            f"Clasificación: <b>riesgo medio</b>. Modalidad: <b>{tipo_proc}</b>.<br>"
            f"Se activaron reglas normativas, pero el puntaje no alcanza el umbral "
            f"de alerta de 71 puntos."
        )
        box_color = "rgba(242, 184, 75, .06)"
        border_c = "#F2B84B"
    else:
        justificacion = (
            f"<b>Evaluación combinada · {pts_total:.1f} de 100 puntos</b><br>"
            f"Transacción dentro de límites normales o sin recurrencia en la ventana configurada. Modalidad: <b>{tipo_proc}</b>."
        )
        box_color = "rgba(139, 124, 255, .06)"
        border_c = "#8B7CFF"

    st.markdown(dedent(f"""
    <div style="background-color: {box_color}; border-left: 3px solid {border_c}; border-radius: 10px; padding: 11px 13px; margin-bottom: 10px; color: #CFD5E2; font-size: .72rem; line-height: 1.5;">
        {justificacion}
    </div>
    """), unsafe_allow_html=True)

    # Desglose legible del puntaje consolidado.
    with st.expander("Desglose del puntaje de riesgo", expanded=True):
        if motivos_reglas:
            for motivo in motivos_reglas:
                st.markdown(f"- {motivo}")
        else:
            st.markdown("- No se activó ninguna regla normativa.")
        st.markdown(
            f"**Subtotal de reglas:** {subtotal_reglas} / 80  \n"
            f"**Ajuste estadístico:** +{pts_bonus_ia}  \n"
            f"**Puntaje consolidado:** {pts_total} / 100 · {banner_txt}"
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.container(border=True):
            encabezado_seccion("Datos de la contratación", icono="file")
            st.write(f"**Objeto de contratación:** {caso_data['descripcion_item']}")
            st.write(f"**Monto referencial:** S/ {float(caso_data['monto_referencial']):,.2f}")
            st.write(f"**Fecha de convocatoria:** {caso_data['fecha_convocatoria']}")
            st.write(f"**Código CUBSO:** {caso_data.get('codigo_cubso', 'GENERAL')}")

    with col2:
        with st.container(border=True):
            encabezado_seccion("Ventana temporal", icono="chart")
            st.write(f"**Monto acumulado en {ventana}:** S/ {float(caso_data['monto_acumulado_30d']):,.2f}")
            st.write(f"**Operaciones en la ventana:** {caso_data['frecuencia_compras_30d']} convocatorias")
            st.write(f"**Separación respecto de la compra previa:** {dias_texto}")
            st.write(f"**Similitud temática:** {float(caso_data['similitud_nlp']):.2%}")

    # Presenta una explicación proporcional cuando no existe un modelo SHAP persistido.
    with st.container(border=True):
        encabezado_seccion(
            "Factores de impacto",
            "Contribución estimada de cada variable al puntaje del caso.",
            "brain",
        )

        df_shap = calcular_explicabilidad_caso(caso_data, ventana=ventana)
        fig_shap = px.bar(
            df_shap,
            x="impacto_porcentaje",
            y="variable_nombre",
            orientation="h",
            text="impacto_porcentaje",
            color="impacto_porcentaje",
            color_continuous_scale=[
                [0.0, "#8B7CFF"],
                [0.55, "#F2B84B"],
                [1.0, "#FF6F87"],
            ],
            labels={
                "impacto_porcentaje": "Contribución al riesgo (%)",
                "variable_nombre": "Variable analizada",
            },
        )
        fig_shap.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_shap.update_layout(coloraxis_showscale=False)
        estilizar_grafica(fig_shap, altura=300, mostrar_leyenda=False)
        st.plotly_chart(
            fig_shap,
            width="stretch",
            config={"displaylogo": False, "responsive": True},
        )

    # La red relaciona operaciones, entidad contratante y categoría CUBSO.
    with st.container(border=True):
        encabezado_seccion(
            "Mapa de relaciones",
            "Vínculos entre la entidad, sus operaciones y las categorías contratadas.",
            "network",
        )

        try:
            df_rel = _cargar_relaciones(lote_id, caso_data['entidad_ruc'])
            fig_net, _ = _construir_grafo_cacheado(
                df_rel,
                caso_data['entidad_ruc'],
            )
            estilizar_grafica(fig_net, altura=390, mostrar_leyenda=False)
            st.plotly_chart(
                fig_net,
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )
        except Exception as e:
            st.error(f"Error al generar grafo de red: {e}")

    # Módulo Human-in-the-Loop y Descarga PDF
    col_h1, col_h2 = st.columns([1, 1])

    with col_h1:
        encabezado_seccion(
            "Decisión de auditoría",
            "Registra el resultado de la revisión y su sustento.",
            "shield",
        )
        
        estados_revision = {
            "Pendiente": "Pendiente",
            "Fraccionamiento confirmado": "Confirmado_Fraccionamiento",
            "Descartado": "Descartado",
        }
        estado_actual = caso_data.get("estado_revision", "Pendiente")
        st.caption(
            f"{etiqueta_estado} · riesgo {nivel.lower()}. "
            f"{explicar_estado_revision(estado_actual, con_auditor)}"
        )
        etiqueta_actual = next(
            (
                etiqueta
                for etiqueta, valor in estados_revision.items()
                if valor == estado_actual
            ),
            "Pendiente",
        )
        nuevo_estado_ui = st.segmented_control(
            "Dictamen final",
            options=list(estados_revision),
            default=etiqueta_actual,
            width="stretch",
        )
        nuevo_estado = estados_revision[nuevo_estado_ui]
        
        comentario_txt = st.text_area(
            "Sustento u observaciones",
            value=caso_data.get('comentario_auditor', '') or '',
        )

        if st.button(
            "Guardar decisión",
            type="primary",
            disabled=not tiene_permiso("dictaminar_caso"),
        ) and tiene_permiso("dictaminar_caso"):
            session = obtener_session()
            try:
                id_usr = st.session_state.get('id_usuario', 1)
                session.execute(text("""
                    UPDATE resultados_anomalias
                    SET estado_revision = :est, comentario_auditor = :com, id_usuario_auditor = :usr, fecha_revision = CURRENT_TIMESTAMP
                    WHERE id_convocatoria = :id_c
                """), {
                    "est": nuevo_estado,
                    "com": comentario_txt,
                    "usr": id_usr,
                    "id_c": caso_data['id_convocatoria']
                })
                session.commit()
                st.cache_data.clear()

                registrar_log(
                    id_usr,
                    "Decisión de auditoría registrada",
                    "Auditoría",
                    (
                        f"Convocatoria {caso_data['id_convocatoria']} "
                        f"dictaminada como {nuevo_estado_ui}."
                    ),
                )
                st.success("La decisión quedó registrada.")
            except Exception as e:
                st.error(f"Error al actualizar dictamen: {e}")
            finally:
                session.close()

    with col_h2:
        encabezado_seccion(
            "Informe de evidencias",
            "Genera un documento formal con los datos y el análisis del expediente.",
            "file",
        )

        if st.button("Generar informe PDF", width="stretch", type="primary"):
            try:
                ruta_pdf = generar_dossier_pdf_auditoria(caso_data, df_shap)
                with open(ruta_pdf, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="Descargar archivo",
                    data=pdf_bytes,
                    file_name=f"Dossier_Evidencia_Conv_{caso_data['id_convocatoria']}.pdf",
                    mime="application/pdf",
                    width="stretch"
                )
            except Exception as e:
                st.error(f"Error al generar PDF: {e}")
