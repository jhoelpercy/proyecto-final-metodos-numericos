"""API REST de SIMTELE-RISK (FastAPI) y servidor de la aplicación web.

Ejecución, desde la raíz del repositorio:
    python -m uvicorn backend.simtele.api:app --port 8000
y abrir http://localhost:8000/
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import VERSION_MODELO, persistencia
from .casos import comun, describir, simular

RAIZ = Path(__file__).resolve().parents[2]
FRONTEND = RAIZ / "frontend"

app = FastAPI(title="SIMTELE-RISK", version=VERSION_MODELO, description="Cuatro casos de estudio de transporte por cable con métodos numéricos. Datos sintéticos.")
_conexion = None
_candado = threading.Lock()


def bd():
    global _conexion
    if _conexion is None:
        _conexion = persistencia.conectar()
    return _conexion


def _simular(modulo, tipo, parametros):
    try:
        return simular(modulo, tipo, parametros)
    except comun.ErrorParametros as e:
        raise HTTPException(status_code=422, detail={"mensaje": "Parámetros no válidos", "problemas": e.problemas})
    except KeyError as e:
        raise HTTPException(status_code=404, detail={"mensaje": str(e)})


@app.get("/api/salud")
def salud():
    return {"estado": "ok", "modelo_version": VERSION_MODELO, "base_de_datos": str(persistencia.RUTA_POR_DEFECTO.name)}


@app.get("/api/modulos")
def modulos():
    return describir()


@app.post("/api/simular")
def api_simular(cuerpo: dict[str, Any] = Body(...)):
    resultado = _simular(cuerpo.get("modulo"), cuerpo.get("tipo", "NORMAL"), cuerpo.get("parametros") or {})
    if cuerpo.get("guardar", True):
        with _candado:
            resultado["simulacion_id"] = persistencia.guardar(bd(), resultado, cuerpo.get("usuario_equipo"))
    return JSONResponse(resultado)


@app.get("/api/simulaciones")
def api_listar(q: str = "", modulo: int | None = None, tipo: str | None = None, limite: int = Query(20, ge=1, le=200), desplazamiento: int = Query(0, ge=0)):
    with _candado:
        return persistencia.listar(bd(), q.strip(), modulo, tipo or None, limite, desplazamiento)


@app.get("/api/simulaciones/{sim_id}")
def api_recuperar(sim_id: int):
    with _candado:
        r = persistencia.recuperar(bd(), sim_id)
    if r is None:
        raise HTTPException(status_code=404, detail={"mensaje": f"No existe la simulación {sim_id}"})
    return JSONResponse(r)


@app.post("/api/simulaciones/{sim_id}/relanzar")
def api_relanzar(sim_id: int, cuerpo: dict[str, Any] = Body(default={})):
    """Recupera los parámetros guardados, repite el cálculo y comprueba la coincidencia dentro de la tolerancia."""
    with _candado:
        original = persistencia.recuperar(bd(), sim_id)
    if original is None:
        raise HTTPException(status_code=404, detail={"mensaje": f"No existe la simulación {sim_id}"})
    nuevo = _simular(original["modulo"], original["tipo"], original["parametros"])
    comparacion = persistencia.comparar(original, nuevo)
    if cuerpo.get("guardar"):
        with _candado:
            nuevo["simulacion_id"] = persistencia.guardar(bd(), nuevo, cuerpo.get("usuario_equipo") or original.get("usuario_equipo"))
    return JSONResponse({"original_id": sim_id, "comparacion": comparacion, "recalculado": nuevo})


@app.get("/estudiantes.json")
def estudiantes():
    return FileResponse(RAIZ / "estudiantes.json", media_type="application/json")


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
