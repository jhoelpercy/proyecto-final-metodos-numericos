"""Pruebas reproducibles con solución analítica de raíces, integración, EDO e interpolación."""

import math

import numpy as np
import pytest

from backend.simtele.numerico import edo, integracion, interpolacion, raices, sistemas

# ---------- raíces ----------
f = lambda x: x * x - 2.0
df = lambda x: 2.0 * x


def test_biseccion_cumple_cota_de_error():
    r = raices.biseccion(f, 0.0, 2.0, tol=1e-12)
    assert r.convergio and abs(r.raiz - math.sqrt(2)) <= 1e-12
    # Cota teórica: (b − a)/2^k
    assert r.iteraciones == math.ceil(math.log2(2.0 / 1e-12))


def test_biseccion_exige_cambio_de_signo():
    with pytest.raises(ValueError):
        raices.biseccion(f, 2.0, 3.0)


def test_newton_converge_cuadraticamente():
    r = raices.newton_raphson(f, df, 1.0, tol=1e-14)
    assert r.convergio and abs(r.raiz - math.sqrt(2)) < 1e-15
    assert r.iteraciones <= 6
    errores = [abs(h["x"] - math.sqrt(2)) for h in r.historial]
    assert errores[2] < errores[1] ** 2 * 1.0  # e_{k+1} ≈ C e_k²


def test_newton_raiz_multiple_es_lineal_con_razon_4_5():
    g = lambda t: (1 - t / 4) ** 5
    dg = lambda t: -5 / 4 * (1 - t / 4) ** 4
    r = raices.newton_raphson(g, dg, 0.0, tol=1e-10, max_iter=500)
    assert r.convergio
    q = raices.orden_convergencia_estimado([h["x"] for h in r.historial], 4.0)
    assert q == pytest.approx(1.0, abs=0.05)
    e = [abs(h["x"] - 4) for h in r.historial]
    assert e[10] / e[9] == pytest.approx(0.8, rel=1e-6)


def test_newton_detecta_derivada_nula():
    r = raices.newton_raphson(lambda x: x * x + 1, lambda x: 2 * x, 0.0)
    assert not r.convergio and "derivada" in r.criterio


def test_secante_superlineal():
    r = raices.secante(f, 1.0, 2.0, tol=1e-14)
    assert r.convergio and abs(r.raiz - math.sqrt(2)) < 1e-15
    assert r.iteraciones <= 8


# ---------- integración ----------
@pytest.mark.parametrize("regla, orden", [(integracion.trapecio, 2), (integracion.simpson, 4)])
def test_orden_de_las_reglas_compuestas(regla, orden):
    e1 = abs(regla(math.sin, 0, math.pi, 16) - 2)
    e2 = abs(regla(math.sin, 0, math.pi, 32) - 2)
    assert math.log2(e1 / e2) == pytest.approx(orden, abs=0.05)


def test_simpson_exacto_para_cubicas_y_romberg_para_quinticas():
    cubica = lambda t: 4 * t**3 - 3 * t**2 + 1
    assert integracion.simpson(cubica, 0, 2, 2) == pytest.approx(16 - 8 + 2, abs=1e-13)
    quintica = lambda t: 6 * t**5
    r = integracion.romberg(quintica, 0, 1, tol=1e-14)
    assert r.convergio and r.valor == pytest.approx(1.0, abs=1e-14)
    assert r.niveles <= 4


def test_romberg_converge_rapido_en_funcion_suave():
    r = integracion.romberg(math.exp, 0, 1, tol=1e-12)
    assert r.convergio and abs(r.valor - (math.e - 1)) < 1e-12
    assert r.evaluaciones < 100


def test_reglas_sobre_muestras():
    t = np.linspace(0, math.pi, 401)
    h = math.pi / 400
    y = np.sin(t)
    assert integracion.trapecio_datos(y, h) == pytest.approx(2, abs=2e-5)
    assert integracion.simpson_datos(y, h) == pytest.approx(2, abs=1e-10)
    assert integracion.romberg_datos(y, h).valor == pytest.approx(2, abs=1e-12)
    # Número impar de subintervalos: Simpson 1/3 + 3/8.
    assert integracion.simpson_datos(y[:400], h) == pytest.approx(1 - math.cos(399 * h), abs=1e-10)


# ---------- EDO ----------
def oscilador(t, y):
    return np.array([y[1], -4.0 * y[0]])


@pytest.mark.parametrize("metodo, orden", [("Heun (RK2)", 2), ("RK4", 4)])
def test_orden_global_de_heun_y_rk4(metodo, orden):
    errores = []
    for h in (0.05, 0.025):
        r = edo.integrar(oscilador, [1.0, 0.0], 0.0, 5.0, h, metodo)
        errores.append(abs(r.y[-1, 0] - math.cos(10.0)))
    assert math.log2(errores[0] / errores[1]) == pytest.approx(orden, abs=0.15)


def test_heun_y_rk4_exactos_con_desaceleracion_constante():
    campo = lambda t, y: np.array([y[1], -2.5])
    for metodo in ("Heun (RK2)", "RK4"):
        r = edo.integrar(campo, [0.0, 5.0], 0.0, 2.0, 0.1, metodo)
        assert r.y[-1, 0] == pytest.approx(5.0, abs=1e-12)
        assert r.y[-1, 1] == pytest.approx(0.0, abs=1e-12)


def test_evento_v_igual_cero_localizado():
    campo = lambda t, y: np.array([y[1], -2.5])
    buscar = lambda g, a, b: raices.secante(g, a, b, 1e-12)
    r, raiz = edo.integrar_hasta_evento(campo, [0.0, 5.0], 0.3, lambda y: y[1], "RK4", buscador=buscar)
    assert r.t[-1] == pytest.approx(2.0, abs=1e-12)
    assert r.y[-1, 0] == pytest.approx(5.0, abs=1e-12)


def test_heun_inestable_con_paso_grande():
    r = edo.integrar(lambda t, y: np.array([y[1], -4 * y[0]]), [1.0, 0.0], 0, 200, 0.5, "Heun (RK2)")
    assert np.max(np.abs(r.y[:, 0])) > 10  # |R(hλ)| > 1: crece sin control físico


# ---------- interpolación ----------
def test_lagrange_y_newton_reproducen_polinomios():
    x = np.array([-1.0, 0.0, 0.5, 2.0, 3.0])
    p = lambda t: 2 * t**4 - t**3 + 0.5 * t - 7
    t = np.linspace(-1, 3, 41)
    coef, _ = interpolacion.diferencias_divididas(x, p(x))
    assert np.max(np.abs(interpolacion.lagrange(x, p(x), t) - p(t))) < 1e-10
    assert np.max(np.abs(interpolacion.newton_evaluar(x, coef, t) - p(t))) < 1e-10
    assert coef[-1] == pytest.approx(2.0)  # coeficiente principal = f[x0..x4]


def test_derivada_de_newton():
    x = np.linspace(0, 1, 5)
    coef, _ = interpolacion.diferencias_divididas(x, x**3)
    assert interpolacion.newton_derivada(x, coef, 0.3)[0] == pytest.approx(3 * 0.09)


def test_spline_natural_interpola_y_es_exacto_para_rectas():
    x = np.array([0.0, 0.7, 1.0, 2.5, 3.0])
    s = interpolacion.spline_natural(x, 3 * x - 1)
    t = np.linspace(0, 3, 31)
    assert np.max(np.abs(s(t) - (3 * t - 1))) < 1e-13
    s2 = interpolacion.spline_natural(x, np.sin(x))
    assert np.allclose(s2(x), np.sin(x), atol=1e-14)
    assert s2.momentos[0] == 0 and s2.momentos[-1] == 0


def test_sistema_del_spline_es_spd_y_lo_resuelven_los_iterativos():
    x = np.linspace(0, 2 * np.pi, 13)
    A, b = interpolacion.sistema_spline_natural(x, np.sin(x))
    assert sistemas.es_spd(A)
    ref = np.linalg.solve(A, b)
    for r in (sistemas.gradiente_conjugado(A, b, 1e-13), sistemas.gauss_seidel(A, b, 1e-13, 500), sistemas.jacobi(A, b, 1e-13, 500)):
        assert r.convergio and np.allclose(r.x, ref, atol=1e-11)


def test_spline_converge_con_orden_cuatro():
    errores = []
    for n in (11, 21):
        x = np.linspace(0, 1, n)
        y = np.exp(x)
        s = interpolacion.spline_natural(x, y)
        t = np.linspace(0.2, 0.8, 101)  # lejos de los extremos (borde natural)
        errores.append(np.max(np.abs(s(t) - np.exp(t))))
    assert math.log2(errores[0] / errores[1]) > 3.5


def test_runge_y_constante_de_lebesgue():
    runge = lambda t: 1 / (1 + 25 * t**2)
    malla = np.linspace(-1, 1, 2001)
    unif = np.linspace(-1, 1, 21)
    cheb = np.cos((2 * np.arange(21) + 1) * np.pi / 42)
    assert interpolacion.constante_lebesgue(unif, malla) > 1e4
    assert interpolacion.constante_lebesgue(cheb, malla) < 3.5
    e_unif = np.max(np.abs(interpolacion.lagrange(unif, runge(unif), malla) - runge(malla)))
    e_cheb = np.max(np.abs(interpolacion.lagrange(cheb, runge(cheb), malla) - runge(malla)))
    assert e_unif > 10 and e_cheb < 0.05
