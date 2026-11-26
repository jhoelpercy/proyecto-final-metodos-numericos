"""El script SQL del modelo de datos crea las 6 entidades con sus restricciones."""

import sqlite3
from pathlib import Path

import pytest

ESQUEMA = Path(__file__).resolve().parents[1] / "schema.sql"
ENTIDADES = {"caso_estudio", "escenario", "simulacion", "ejecucion_metodo", "iteracion_resultado", "resultado_resumen"}


@pytest.fixture
def bd():
    conexion = sqlite3.connect(":memory:")
    conexion.executescript(ESQUEMA.read_text(encoding="utf-8"))
    yield conexion
    conexion.close()


def test_seis_entidades(bd):
    tablas = {f[0] for f in bd.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tablas == ENTIDADES


def test_cadena_completa_y_borrado_en_cascada(bd):
    bd.executescript(
        """
        INSERT INTO caso_estudio VALUES (1, 'Deterioro', '1.0');
        INSERT INTO escenario VALUES (1, 1, 'NORMAL', '{"n": 6}', 'Sintético');
        INSERT INTO simulacion (id, escenario_id, fecha, usuario_equipo, estado) VALUES (1, 1, '2026-11-06', 'Equipo', 'COMPLETADA');
        INSERT INTO ejecucion_metodo (id, simulacion_id, metodo, tolerancia, paso, max_iter, convergio, tiempo_ms) VALUES (1, 1, 'Jacobi', 1e-10, NULL, 500, 1, 0.5);
        INSERT INTO iteracion_resultado VALUES (1, 1, 1, '{"x": [1.0]}', 0.1, 0.1, 0.1);
        INSERT INTO resultado_resumen VALUES (1, 1, 'x*', '{"x": [1.0]}', '{"error_abs": 0.1}', 'Prueba');
        DELETE FROM simulacion WHERE id = 1;
        """
    )
    for tabla in ("ejecucion_metodo", "iteracion_resultado", "resultado_resumen"):
        assert bd.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0] == 0


@pytest.mark.parametrize(
    "sentencia",
    [
        "INSERT INTO escenario VALUES (1, 99, 'NORMAL', '{}', 'x')",
        "INSERT INTO caso_estudio VALUES (NULL, NULL, '1.0')",
    ],
)
def test_integridad_referencial(bd, sentencia):
    with pytest.raises(sqlite3.IntegrityError):
        bd.execute(sentencia)


def test_tipo_de_escenario_restringido(bd):
    bd.execute("INSERT INTO caso_estudio VALUES (1, 'Caso', '1.0')")
    with pytest.raises(sqlite3.IntegrityError):
        bd.execute("INSERT INTO escenario VALUES (1, 1, 'OTRO', '{}', 'x')")


def test_resultado_json_debe_ser_valido(bd):
    bd.execute("INSERT INTO caso_estudio VALUES (1, 'Caso', '2.0')")
    bd.execute("INSERT INTO escenario VALUES (1, 1, 'NORMAL', '{}', 'x')")
    with pytest.raises(sqlite3.IntegrityError):
        bd.execute("INSERT INTO simulacion (escenario_id, fecha, estado, resultado_json) VALUES (1, 'hoy', 'COMPLETADA', '{no json')")
