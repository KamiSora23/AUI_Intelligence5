import unittest

import numpy as np
import pandas as pd

from evaluacion_rigurosa import (
    VARIABLES_IF_BASE,
    composicion_etiqueta,
    con_contexto,
    intervalo_95,
    isolation_forest_solo,
    metricas,
    por_entidad,
    triaje,
    validacion_anidada,
)


class EvaluacionRigurosaTest(unittest.TestCase):
    def test_metricas_de_la_matriz_de_confusion(self):
        y = np.array([1, 1, 1, 0, 0])
        m = metricas(y, np.array([1, 1, 0, 1, 0]), np.array([90, 80, 10, 75, 5]))
        self.assertEqual((m["VP"], m["FP"], m["FN"], m["VN"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["sensibilidad"], 2 / 3)
        self.assertAlmostEqual(m["especificidad"], 0.5)
        self.assertAlmostEqual(m["exactitud_balanceada"], (2 / 3 + 0.5) / 2)

    def test_validacion_anidada_no_mira_la_entidad_evaluada(self):
        # La entidad "X" es la única donde la similitud 0.60 ayuda: al dejarla
        # fuera, el resto no lo muestra y debe elegirse el valor por defecto.
        entidades = np.array(["X", "X", "A", "A", "B", "B"])
        y = np.array([1, 0, 1, 0, 1, 0])
        puntajes = {
            0.50: np.array([80, 80, 80, 0, 80, 0]),  # en X marca el conforme
            0.60: np.array([80, 0, 80, 0, 80, 0]),   # en X lo descarta
        }
        decision, _, elegidas = validacion_anidada(y, entidades, puntajes, 71)
        self.assertEqual(elegidas, {0.50: 1, 0.60: 2})
        # X se evalúa con 0.50: su conforme sigue siendo falsa alarma.
        self.assertEqual(decision[entidades == "X"].tolist(), [True, True])

    def test_intervalo_es_reproducible_y_contiene_el_valor(self):
        rng = np.random.default_rng(0)
        entidades = np.repeat(np.arange(20).astype(str), 5)
        y = rng.integers(0, 2, 100)
        decision = y.astype(bool) ^ (rng.uniform(size=100) < 0.2)
        a = intervalo_95(y, decision, entidades, "exactitud_balanceada", 500)
        b = intervalo_95(y, decision, entidades, "exactitud_balanceada", 500)
        self.assertEqual(a, b)
        observado = metricas(y, decision, decision.astype(float))["exactitud_balanceada"]
        self.assertLessEqual(a[0], observado)
        self.assertGreaterEqual(a[1], observado)

    def test_medidas_por_entidad_y_con_contexto(self):
        # A: dos compras con fraccionamiento, una detectada · B: compra única
        # con fraccionamiento, sin alerta · C: conforme, sin alerta.
        entidades = np.array(["A", "A", "B", "C", "C"])
        y = np.array([1, 1, 1, 0, 0])
        puntaje = np.array([80, 55, 0, 10, 0])

        composicion = composicion_etiqueta(y, entidades).set_index("composicion")["entidades"]
        self.assertEqual(composicion.to_dict(),
                         {"Todas con fraccionamiento": 2, "Todas conformes": 1, "Mixta": 0})

        entidad = por_entidad(y, entidades, puntaje, 71).set_index("criterio")
        alto = entidad.loc["Alguna compra en riesgo alto"]
        self.assertEqual((alto["detectadas"], alto["entidades_con_fraccionamiento"]), (1, 2))
        self.assertEqual(alto["conformes_con_alerta"], 0)

        contexto = con_contexto(y, entidades, puntaje, 71).set_index("grupo")
        multiples = contexto.loc["Entidades con más de una compra"]
        self.assertEqual((multiples["fraccionamientos"], multiples["detectados_alto"],
                          multiples["alto_o_pendiente"]), (2, 1, 2))
        self.assertEqual(contexto.loc["Entidades con una sola compra", "detectados_alto"], 0)

    def test_isolation_forest_solo_marca_lo_atipico_sin_nlp(self):
        # 19 compras parecidas y una muy distinta en todas las variables; la
        # similitud semántica no debe influir en la decisión.
        rng = np.random.default_rng(0)
        datos = pd.DataFrame({c: rng.normal(100, 5, 20) for c in VARIABLES_IF_BASE})
        datos.loc[19, list(VARIABLES_IF_BASE)] = 5000
        datos["max_similitud_nlp"] = 1.0
        decision, puntaje = isolation_forest_solo(datos, contaminacion=0.05)
        self.assertEqual(np.flatnonzero(decision).tolist(), [19])
        self.assertEqual(int(np.argmax(puntaje)), 19)
        datos["max_similitud_nlp"] = 0.0
        self.assertEqual(isolation_forest_solo(datos, 0.05)[0].tolist(), decision.tolist())

    def test_triaje_reparte_todos_los_casos(self):
        y = np.array([1, 1, 0, 0, 1])
        tabla = triaje(y, np.array([80, 55, 55, 10, 0]), 71)
        self.assertEqual(tabla["propuesta"].tolist(),
                         ["Propuesta de confirmar", "Pendiente de revisión", "Propuesta de descartar"])
        self.assertEqual(tabla[["fraccionamiento", "conformes"]].to_numpy().sum(), len(y))


if __name__ == "__main__":
    unittest.main()
