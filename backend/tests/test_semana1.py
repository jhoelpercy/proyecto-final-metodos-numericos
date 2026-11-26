"""Teoría de errores y sistemas lineales SPD con solución de referencia."""

import json

import numpy as np
import pytest

from backend.simtele import experimentos
from backend.simtele.modulos import MODULOS
from backend.simtele.numerico import errores, sistemas


def test_errores_a_mano():
    assert errores.error_absoluto(2.0, 1.5) == 0.5
    assert errores.error_relativo(2.0, 1.5) == 0.25
    assert errores.error_porcentual(2.0, 1.5) == 25.0
    assert errores.error_relativo(0.0, 1.0) is None
    assert errores.error_absoluto_vector([3.0, 4.0], [0.0, 0.0]) == 5.0
    assert errores.residual_lineal(np.eye(2), np.ones(2), np.ones(2)) == 0.0
    assert errores.numero_condicion_relativo(1.0, 1.0 + 1e-7, 1.0, (1.0 + 1e-7) ** 2) == pytest.approx(2.0, abs=1e-6)


class TestEjemploManual4x4:
    """Sistema del informe: A tridiagonal con diagonal 4 y x* = (1, -2, 3, 0.5)."""

    A = np.array([[4.0, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]])
    x = np.array([1.0, -2.0, 3.0, 0.5])
    b = A @ x

    def test_b_a_mano(self):
        assert self.b == pytest.approx([2.0, -4.0, 10.5, 5.0])

    def test_primer_barrido_jacobi_a_mano(self):
        r = sistemas.jacobi(self.A, self.b, 1e-30, 1)
        assert r.x == pytest.approx([0.5, -1.0, 2.625, 1.25])

    def test_primer_barrido_gauss_seidel_a_mano(self):
        r = sistemas.gauss_seidel(self.A, self.b, 1e-30, 1)
        assert r.x == pytest.approx([0.5, -1.125, 2.90625, 0.5234375])

    @pytest.mark.parametrize("metodo", [sistemas.jacobi, sistemas.gauss_seidel, sistemas.gradiente_conjugado])
    def test_iterativos_convergen(self, metodo):
        r = metodo(self.A, self.b, 1e-12)
        assert r.convergio and r.x == pytest.approx(self.x, abs=1e-10)

    def test_lu(self):
        L, U, P = sistemas.factorizar_lu(self.A)
        assert L @ U == pytest.approx(self.A[P])
        assert sistemas.resolver_lu(self.A, self.b).x == pytest.approx(self.x)

    def test_gradiente_conjugado_a_lo_sumo_n_pasos(self):
        assert sistemas.gradiente_conjugado(self.A, self.b, 1e-10).iteraciones <= 4


def test_lu_singular_y_pivoteo():
    with pytest.raises(np.linalg.LinAlgError):
        sistemas.factorizar_lu(np.array([[1.0, 2.0], [2.0, 4.0]]))
    A = np.array([[0.0, 2.0], [3.0, 1.0]])
    assert sistemas.resolver_lu(A, np.array([4.0, 5.0])).x == pytest.approx(np.linalg.solve(A, np.array([4.0, 5.0])))


def test_todas_las_matrices_son_spd_y_de_tamano_minimo():
    for tipo in experimentos.ESCENARIOS:
        A, _, _ = experimentos.construir_sistema(tipo)
        assert sistemas.es_spd(A) and A.shape[0] >= 4


@pytest.fixture(scope="module")
def resultados():
    return {e["tipo"]: e for e in experimentos.generar()["sistemas"]}


def test_normal_kappa_cercano_a_uno(resultados):
    assert resultados["NORMAL"]["kappa"] < 2
    assert all(m["convergio"] for m in resultados["NORMAL"]["metodos"])


def test_mal_condicionado_evidencia(resultados):
    e = resultados["MAL_CONDICIONADO"]
    assert e["kappa"] > 1000 and e["rho_jacobi"] > 1
    por_metodo = {m["metodo"]: m for m in e["metodos"]}
    assert not por_metodo["Jacobi"]["convergio"]
    lu = por_metodo["LU con pivoteo parcial"]
    assert lu["error_abs"] > 1e3 * lu["residual"]


def test_perturbacion_acotada_por_kappa_y_grande_en_hilbert(resultados):
    for e in resultados.values():
        assert e["amplificacion_perturbacion"] <= 1.1 * e["kappa"]
    assert resultados["MAL_CONDICIONADO"]["amplificacion_perturbacion"] > 1e9


def test_datos_exportables_sin_nan():
    json.dumps(experimentos.generar(), allow_nan=False)


def test_definicion_de_los_cuatro_modulos():
    assert [m["codigo"] for m in MODULOS] == ["detencion", "deterioro", "oscilaciones", "sensores"]
    for m in MODULOS:
        assert set(m["escenarios"]) == {"NORMAL", "ESTRES", "MAL_CONDICIONADO"}
        assert m["referencia"] and m["hipotesis"] and m["metodos"]
