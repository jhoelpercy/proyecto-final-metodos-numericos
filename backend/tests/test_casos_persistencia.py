"""Los 12 escenarios, la persistencia (guardar → reiniciar → recuperar → relanzar) y la API."""

import json
import math

import pytest

from backend.simtele import persistencia
from backend.simtele.casos import CASOS, comun, simular


@pytest.fixture(scope="module")
def resultados():
    return {(codigo, tipo): simular(codigo, tipo) for codigo, caso in CASOS.items() for tipo in caso.ESCENARIOS}


def test_doce_escenarios_con_evidencias(resultados):
    assert len(resultados) == 12
    for (codigo, tipo), r in resultados.items():
        json.dumps(r, allow_nan=False)
        assert len(r["graficos"]) >= 2, (codigo, tipo)
        assert r["tablas"] and r["ejecuciones"] and r["indicadores"]
        assert r["conclusiones"]["numerica"] and r["conclusiones"]["problema"]
        assert r["es_subexperimento"] == (tipo == "MAL_CONDICIONADO")
        for e in r["ejecuciones"]:
            assert e["resumen"]["referencia"] and e["iteraciones"], (codigo, tipo, e["etiqueta"])


def test_todas_las_familias_estan_cubiertas(resultados):
    familias = {e["familia"] for r in resultados.values() for e in r["ejecuciones"]}
    assert familias == {"EDO", "Raíces", "Integración", "Sistemas lineales", "Interpolación"}
    metodos = {e["etiqueta"] for r in resultados.values() for e in r["ejecuciones"]}
    for m in ["Heun (RK2)", "RK4", "Bisección", "Newton-Raphson", "Secante", "Trapecio compuesto", "Simpson 1/3 compuesto", "Romberg",
              "Jacobi", "Gauss-Seidel", "LU con pivoteo parcial", "Gradiente conjugado", "Lagrange", "Newton (diferencias divididas)", "Spline cúbico natural"]:
        assert m in metodos


def _ej(r, nombre):
    return next(e for e in r["ejecuciones"] if e["etiqueta"] == nombre)


def test_m1_valores_de_referencia(resultados):
    n = resultados[("detencion", "NORMAL")]
    rk4 = _ej(n, "RK4")["resumen"]["valor"]
    assert rk4["t_det"] == pytest.approx(2.0, abs=1e-12) and rk4["d_total"] == pytest.approx(7.5, abs=1e-12)
    e = resultados[("detencion", "ESTRES")]
    d_ref = math.log1p(0.05 * 49 / 1.2) / 0.1
    assert _ej(e, "RK4")["resumen"]["valor"]["d_fren"] == pytest.approx(d_ref, abs=1e-3)
    assert _ej(e, "Heun (RK2)")["resumen"]["metricas"]["error_abs_d_fren"] > _ej(e, "RK4")["resumen"]["metricas"]["error_abs_d_fren"]
    m = resultados[("detencion", "MAL_CONDICIONADO")]
    assert _ej(m, "Newton-Raphson")["total_iteraciones"] > 50  # raíz múltiple: convergencia lineal


def test_m2_raiz_y_sistemas(resultados):
    n = resultados[("deterioro", "NORMAL")]
    t = (-0.8 + math.sqrt(0.64 + 4 * 0.02 * 30)) / 0.04
    assert _ej(n, "Newton-Raphson")["resumen"]["valor"]["raiz"] == pytest.approx(t, abs=1e-10)
    m = resultados[("deterioro", "MAL_CONDICIONADO")]
    assert not _ej(m, "Jacobi")["convergio"]
    lu = _ej(m, "LU con pivoteo parcial")["resumen"]["metricas"]
    assert lu["error_rel"] > 1e6 * lu["residual"]  # residuo pequeño ≠ error pequeño


def test_m3_errores_y_estabilidad(resultados):
    assert _ej(resultados[("oscilaciones", "NORMAL")], "RK4")["resumen"]["metricas"]["error_max_x"] < 1e-5
    e = resultados[("oscilaciones", "ESTRES")]
    assert _ej(e, "Heun (RK2)")["resumen"]["metricas"]["factor_amplificacion"] > 1
    assert _ej(e, "RK4")["resumen"]["metricas"]["factor_amplificacion"] < 1


def test_m4_runge_y_spline(resultados):
    m = resultados[("sensores", "MAL_CONDICIONADO")]
    assert _ej(m, "Lagrange")["resumen"]["metricas"]["error_max_malla"] > 10
    assert _ej(m, "Spline cúbico natural")["resumen"]["metricas"]["error_max_malla"] < 0.01
    n = resultados[("sensores", "NORMAL")]
    assert _ej(n, "Lagrange")["resumen"]["metricas"]["error_max_oculto"] < 1e-6


def test_validacion_de_parametros():
    with pytest.raises(comun.ErrorParametros) as e:
        simular("detencion", "NORMAL", {"v0": -3, "h": "x"})
    assert len(e.value.problemas) == 2
    with pytest.raises(comun.ErrorParametros):
        simular("sensores", "NORMAL", {"senal": "cuadrada"})
    with pytest.raises(comun.ErrorParametros):
        simular("deterioro", "NORMAL", {"matriz": "hilbert", "n": 20})


def test_guardar_cerrar_recuperar_y_repetir(tmp_path, resultados):
    """Prueba obligatoria: ejecutar → guardar → cerrar → recuperar → repetir → coincidir."""
    ruta = tmp_path / "historial.db"
    original = resultados[("deterioro", "ESTRES")]
    con = persistencia.conectar(ruta)
    sim_id = persistencia.guardar(con, original, "Equipo de prueba")
    con.close()  # «reinicio» de la aplicación

    con = persistencia.conectar(ruta)
    recuperado = persistencia.recuperar(con, sim_id)
    assert recuperado["parametros"] == original["parametros"]
    assert recuperado["graficos"] == original["graficos"] and recuperado["tablas"] == original["tablas"]
    assert [e["iteraciones"] for e in recuperado["ejecuciones"]] == [e["iteraciones"] for e in original["ejecuciones"]]
    repetido = simular(recuperado["modulo"], recuperado["tipo"], recuperado["parametros"])
    informe = persistencia.comparar(recuperado, repetido)
    assert informe["coincide"] and informe["valores_comparados"] > 1000
    assert persistencia.listar(con, "prueba")["total"] == 1
    con.close()


def test_comparar_detecta_diferencias(resultados):
    a = resultados[("detencion", "NORMAL")]
    b = json.loads(json.dumps(a))
    b["ejecuciones"][1]["resumen"]["valor"]["d_fren"] *= 1 + 1e-6
    assert not persistencia.comparar(a, b)["coincide"]


def test_api_extremo_a_extremo(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.simtele import api

    monkeypatch.setattr(persistencia, "RUTA_POR_DEFECTO", tmp_path / "api.db")
    monkeypatch.setattr(api, "_conexion", None)
    cliente = TestClient(api.app)
    assert cliente.get("/api/salud").json()["estado"] == "ok"
    assert len(cliente.get("/api/modulos").json()) == 4
    r = cliente.post("/api/simular", json={"modulo": "sensores", "tipo": "ESTRES", "usuario_equipo": "API"})
    assert r.status_code == 200
    sim_id = r.json()["simulacion_id"]
    assert cliente.post("/api/simular", json={"modulo": "detencion", "parametros": {"v0": 99}}).status_code == 422
    assert cliente.get("/api/simulaciones", params={"modulo": 4}).json()["total"] == 1
    assert cliente.get(f"/api/simulaciones/{sim_id}").json()["simulacion_id"] == sim_id
    relanzado = cliente.post(f"/api/simulaciones/{sim_id}/relanzar", json={"guardar": True}).json()
    assert relanzado["comparacion"]["coincide"] and relanzado["recalculado"]["simulacion_id"] == sim_id + 1
    assert cliente.get("/api/simulaciones/9999").status_code == 404
