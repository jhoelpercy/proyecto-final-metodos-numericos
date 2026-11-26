-- SIMTELE-RISK: modelo de datos (SQLite 3). Versión 2 (semana 2).
-- Cambios respecto de la semana 1:
--   * simulacion.resultado_json guarda tablas, gráficos, indicadores y conclusiones del escenario;
--   * simulacion.tiempo_total_ms y ejecucion_metodo.familia / criterio / orden completan la trazabilidad.
-- Las iteraciones y resúmenes de cada método siguen normalizados en sus propias tablas.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS caso_estudio (
    id             INTEGER PRIMARY KEY,
    nombre         TEXT NOT NULL UNIQUE,
    modelo_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escenario (
    id              INTEGER PRIMARY KEY,
    caso_id         INTEGER NOT NULL REFERENCES caso_estudio(id) ON DELETE RESTRICT,
    tipo            TEXT NOT NULL CHECK (tipo IN ('NORMAL', 'ESTRES', 'MAL_CONDICIONADO')),
    parametros_json TEXT NOT NULL CHECK (json_valid(parametros_json)),
    fuente_datos    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulacion (
    id              INTEGER PRIMARY KEY,
    escenario_id    INTEGER NOT NULL REFERENCES escenario(id) ON DELETE RESTRICT,
    fecha           TEXT NOT NULL,
    usuario_equipo  TEXT,
    estado          TEXT NOT NULL CHECK (estado IN ('COMPLETADA', 'FALLIDA')),
    resultado_json  TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(resultado_json)),
    tiempo_total_ms REAL CHECK (tiempo_total_ms IS NULL OR tiempo_total_ms >= 0)
);

CREATE TABLE IF NOT EXISTS ejecucion_metodo (
    id            INTEGER PRIMARY KEY,
    simulacion_id INTEGER NOT NULL REFERENCES simulacion(id) ON DELETE CASCADE,
    metodo        TEXT NOT NULL,
    tolerancia    REAL,
    paso          REAL,
    max_iter      INTEGER,
    convergio     INTEGER NOT NULL CHECK (convergio IN (0, 1)),
    tiempo_ms     REAL NOT NULL CHECK (tiempo_ms >= 0),
    familia       TEXT NOT NULL DEFAULT 'Sistemas lineales',
    criterio      TEXT,
    orden         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS iteracion_resultado (
    id                INTEGER PRIMARY KEY,
    ejecucion_id      INTEGER NOT NULL REFERENCES ejecucion_metodo(id) ON DELETE CASCADE,
    numero            INTEGER NOT NULL,
    aproximacion_json TEXT NOT NULL CHECK (json_valid(aproximacion_json)),
    residual          REAL,
    error_abs         REAL,
    error_rel         REAL
);

CREATE TABLE IF NOT EXISTS resultado_resumen (
    id                 INTEGER PRIMARY KEY,
    ejecucion_id       INTEGER NOT NULL UNIQUE REFERENCES ejecucion_metodo(id) ON DELETE CASCADE,
    referencia         TEXT NOT NULL,
    valor_json         TEXT NOT NULL CHECK (json_valid(valor_json)),
    metrica_error_json TEXT NOT NULL CHECK (json_valid(metrica_error_json)),
    interpretacion     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_escenario_caso ON escenario(caso_id, tipo);
CREATE INDEX IF NOT EXISTS ix_simulacion_escenario ON simulacion(escenario_id);
CREATE INDEX IF NOT EXISTS ix_simulacion_fecha ON simulacion(fecha);
CREATE INDEX IF NOT EXISTS ix_ejecucion_simulacion ON ejecucion_metodo(simulacion_id);
CREATE INDEX IF NOT EXISTS ix_iteracion_ejecucion ON iteracion_resultado(ejecucion_id, numero);
