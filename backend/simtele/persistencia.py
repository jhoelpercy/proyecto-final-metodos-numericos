"""Persistencia en SQLite: guardar, listar, recuperar y comparar simulaciones completas.

Correspondencia con el modelo de datos (database/schema.sql):
    resultado canónico            → simulacion (+ resultado_json con tablas, gráficos y conclusiones)
    resultado["parametros"]       → escenario.parametros_json (se reutiliza si ya existe idéntico)
    resultado["ejecuciones"][i]   → ejecucion_metodo
        ["iteraciones"][k]        → iteracion_resultado
        ["resumen"]               → resultado_resumen
El historial vive en un archivo .db, no en memoria: sobrevive al reinicio de la aplicación.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
ESQUEMA = RAIZ / "database" / "schema.sql"
RUTA_POR_DEFECTO = Path(os.environ.get("SIMTELE_DB", RAIZ / "database" / "simtele.db"))

CLAVES_DOCUMENTO = ("descripcion", "indicadores", "tablas", "graficos", "conclusiones", "panel", "es_subexperimento", "nombre_modulo", "modelo_version")
TOLERANCIA_REL = 1e-9
TOLERANCIA_ABS = 1e-12


def _json(valor) -> str:
    return json.dumps(valor, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def conectar(ruta: str | Path | None = None) -> sqlite3.Connection:
    ruta = Path(ruta) if ruta else RUTA_POR_DEFECTO
    if str(ruta) != ":memory:":
        ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(ruta), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(ESQUEMA.read_text(encoding="utf-8"))
    return con


def guardar(con: sqlite3.Connection, resultado: dict, usuario_equipo: str | None = None) -> int:
    """Guarda una simulación completa en una sola transacción y devuelve su id."""
    with con:
        con.execute("INSERT OR IGNORE INTO caso_estudio (id, nombre, modelo_version) VALUES (?, ?, ?)", (resultado["modulo_id"], resultado["nombre_modulo"], resultado["modelo_version"]))
        con.execute("UPDATE caso_estudio SET modelo_version = ? WHERE id = ?", (resultado["modelo_version"], resultado["modulo_id"]))
        parametros = _json(resultado["parametros"])
        fila = con.execute("SELECT id FROM escenario WHERE caso_id = ? AND tipo = ? AND parametros_json = ?", (resultado["modulo_id"], resultado["tipo"], parametros)).fetchone()
        if fila:
            escenario_id = fila["id"]
        else:
            escenario_id = con.execute(
                "INSERT INTO escenario (caso_id, tipo, parametros_json, fuente_datos) VALUES (?, ?, ?, ?)",
                (resultado["modulo_id"], resultado["tipo"], parametros, resultado["fuente_datos"]),
            ).lastrowid
        documento = {k: resultado.get(k) for k in CLAVES_DOCUMENTO}
        estado = "COMPLETADA" if resultado.get("ejecuciones") else "FALLIDA"
        sim_id = con.execute(
            "INSERT INTO simulacion (escenario_id, fecha, usuario_equipo, estado, resultado_json, tiempo_total_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (escenario_id, datetime.now().isoformat(timespec="seconds"), (usuario_equipo or "").strip()[:80] or None, estado, _json(documento), resultado.get("tiempo_total_ms")),
        ).lastrowid
        for orden, e in enumerate(resultado["ejecuciones"]):
            ej_id = con.execute(
                "INSERT INTO ejecucion_metodo (simulacion_id, metodo, tolerancia, paso, max_iter, convergio, tiempo_ms, familia, criterio, orden) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sim_id, e["etiqueta"], e["tolerancia"], e["paso"], e["max_iter"], int(e["convergio"]), max(0.0, e["tiempo_ms"] or 0.0), e["familia"], e["criterio"], orden),
            ).lastrowid
            con.executemany(
                "INSERT INTO iteracion_resultado (ejecucion_id, numero, aproximacion_json, residual, error_abs, error_rel) VALUES (?, ?, ?, ?, ?, ?)",
                [(ej_id, it["numero"], _json(it["aproximacion"]), it["residual"], it["error_abs"], it["error_rel"]) for it in e["iteraciones"]],
            )
            r = e["resumen"]
            con.execute(
                "INSERT INTO resultado_resumen (ejecucion_id, referencia, valor_json, metrica_error_json, interpretacion) VALUES (?, ?, ?, ?, ?)",
                (ej_id, r["referencia"], _json(r["valor"]), _json({**r["metricas"], "_total_iteraciones": e["total_iteraciones"]}), r["interpretacion"]),
            )
    return int(sim_id)


def listar(con: sqlite3.Connection, texto: str = "", modulo: int | None = None, tipo: str | None = None, limite: int = 50, desplazamiento: int = 0) -> dict:
    condiciones, args = [], []
    if modulo:
        condiciones.append("c.id = ?")
        args.append(int(modulo))
    if tipo:
        condiciones.append("e.tipo = ?")
        args.append(tipo)
    if texto:
        condiciones.append("(c.nombre LIKE ? OR IFNULL(s.usuario_equipo, '') LIKE ? OR e.parametros_json LIKE ? OR CAST(s.id AS TEXT) = ? OR s.fecha LIKE ?)")
        patron = f"%{texto}%"
        args += [patron, patron, patron, texto.strip("#"), patron]
    donde = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
    base = f"FROM simulacion s JOIN escenario e ON e.id = s.escenario_id JOIN caso_estudio c ON c.id = e.caso_id {donde}"
    total = con.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
    filas = con.execute(
        f"""SELECT s.id, s.fecha, s.usuario_equipo, s.estado, s.tiempo_total_ms, e.tipo, e.parametros_json, c.id AS modulo_id, c.nombre,
               (SELECT COUNT(*) FROM ejecucion_metodo em WHERE em.simulacion_id = s.id) AS ejecuciones
            {base} ORDER BY s.id DESC LIMIT ? OFFSET ?""",
        args + [int(limite), int(desplazamiento)],
    ).fetchall()
    return {
        "total": total,
        "simulaciones": [
            {"id": f["id"], "fecha": f["fecha"], "usuario_equipo": f["usuario_equipo"], "estado": f["estado"], "tipo": f["tipo"], "modulo_id": f["modulo_id"],
             "nombre_modulo": f["nombre"], "parametros": json.loads(f["parametros_json"]), "ejecuciones": f["ejecuciones"], "tiempo_total_ms": f["tiempo_total_ms"]}
            for f in filas
        ],
    }


def recuperar(con: sqlite3.Connection, sim_id: int) -> dict | None:
    """Reconstruye el resultado canónico completo leyendo sólo de la base de datos."""
    s = con.execute(
        """SELECT s.*, e.tipo, e.parametros_json, e.fuente_datos, c.id AS modulo_id, c.nombre, c.modelo_version AS version_caso
           FROM simulacion s JOIN escenario e ON e.id = s.escenario_id JOIN caso_estudio c ON c.id = e.caso_id WHERE s.id = ?""",
        (int(sim_id),),
    ).fetchone()
    if s is None:
        return None
    documento = json.loads(s["resultado_json"])
    ejecuciones = []
    for em in con.execute("SELECT * FROM ejecucion_metodo WHERE simulacion_id = ? ORDER BY orden, id", (sim_id,)):
        rr = con.execute("SELECT * FROM resultado_resumen WHERE ejecucion_id = ?", (em["id"],)).fetchone()
        metricas = json.loads(rr["metrica_error_json"])
        total = metricas.pop("_total_iteraciones", None)
        iteraciones = [
            {"numero": it["numero"], "aproximacion": json.loads(it["aproximacion_json"]), "residual": it["residual"], "error_abs": it["error_abs"], "error_rel": it["error_rel"]}
            for it in con.execute("SELECT * FROM iteracion_resultado WHERE ejecucion_id = ? ORDER BY id", (em["id"],))
        ]
        ejecuciones.append({
            "metodo": em["metodo"], "etiqueta": em["metodo"], "familia": em["familia"], "tolerancia": em["tolerancia"], "paso": em["paso"], "max_iter": em["max_iter"],
            "convergio": bool(em["convergio"]), "criterio": em["criterio"], "tiempo_ms": em["tiempo_ms"], "iteraciones": iteraciones,
            "total_iteraciones": total if total is not None else len(iteraciones),
            "resumen": {"referencia": rr["referencia"], "valor": json.loads(rr["valor_json"]), "metricas": metricas, "interpretacion": rr["interpretacion"]},
        })
    from .casos import POR_ID  # import diferido: evita ciclos

    caso = POR_ID[s["modulo_id"]]
    return {
        **documento,
        "modulo": caso.CODIGO,
        "modulo_id": s["modulo_id"],
        "nombre_modulo": s["nombre"],
        "tipo": s["tipo"],
        "parametros": json.loads(s["parametros_json"]),
        "fuente_datos": s["fuente_datos"],
        "ejecuciones": ejecuciones,
        "tiempo_total_ms": s["tiempo_total_ms"],
        "simulacion_id": s["id"],
        "fecha": s["fecha"],
        "usuario_equipo": s["usuario_equipo"],
        "estado": s["estado"],
    }


def _numeros(valor, ruta=""):
    """Aplana estructuras a pares (ruta, número) para comparar."""
    if isinstance(valor, bool) or valor is None or isinstance(valor, str):
        return
    if isinstance(valor, (int, float)):
        yield ruta, float(valor)
    elif isinstance(valor, dict):
        for k, v in valor.items():
            yield from _numeros(v, f"{ruta}.{k}")
    elif isinstance(valor, list):
        for i, v in enumerate(valor):
            yield from _numeros(v, f"{ruta}[{i}]")


def comparar(original: dict, recalculado: dict, rtol: float = TOLERANCIA_REL, atol: float = TOLERANCIA_ABS) -> dict:
    """Compara método a método valores, métricas e iteraciones (se excluyen los tiempos de cómputo)."""
    detalles, max_dif, total, fallas = [], 0.0, 0, 0
    pares = list(zip(original["ejecuciones"], recalculado["ejecuciones"]))
    if len(original["ejecuciones"]) != len(recalculado["ejecuciones"]):
        fallas += 1
    for a, b in pares:
        dif_max, n, fuera = 0.0, 0, 0
        izquierda = {"valor": a["resumen"]["valor"], "metricas": a["resumen"]["metricas"], "iter": [[i["numero"], i["aproximacion"], i["residual"], i["error_abs"], i["error_rel"]] for i in a["iteraciones"]]}
        derecha = {"valor": b["resumen"]["valor"], "metricas": b["resumen"]["metricas"], "iter": [[i["numero"], i["aproximacion"], i["residual"], i["error_abs"], i["error_rel"]] for i in b["iteraciones"]]}
        dl, dr = dict(_numeros(izquierda)), dict(_numeros(derecha))
        claves = set(dl) | set(dr)
        for k in claves:
            if k not in dl or k not in dr:
                fuera += 1
                continue
            x, y = dl[k], dr[k]
            if not (math.isfinite(x) and math.isfinite(y)):
                fuera += x != y
                continue
            d = abs(x - y)
            escala = max(abs(x), abs(y))
            dif_max = max(dif_max, d / escala if escala > 0 else d)
            n += 1
            if d > atol + rtol * escala:
                fuera += 1
        coincide = fuera == 0 and a["convergio"] == b["convergio"] and a["etiqueta"] == b["etiqueta"]
        fallas += 0 if coincide else 1
        total += n
        max_dif = max(max_dif, dif_max)
        detalles.append({"metodo": a["etiqueta"], "familia": a["familia"], "valores_comparados": n, "fuera_de_tolerancia": fuera, "max_diferencia_relativa": dif_max, "coincide": coincide})
    return {"coincide": fallas == 0, "valores_comparados": total, "max_diferencia_relativa": max_dif, "rtol": rtol, "atol": atol, "detalles": detalles}
