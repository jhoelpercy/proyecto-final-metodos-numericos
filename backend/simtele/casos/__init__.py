"""Registro de los cuatro casos de estudio y punto de entrada único para simular."""

from __future__ import annotations

from .. import VERSION_MODELO
from ..modulos import MODULOS
from . import comun, deterioro, detencion, oscilaciones, sensores

CASOS = {m.CODIGO: m for m in (detencion, deterioro, oscilaciones, sensores)}
POR_ID = {m.ID: m for m in CASOS.values()}


def obtener(modulo) -> object:
    if isinstance(modulo, int) or (isinstance(modulo, str) and modulo.isdigit()):
        caso = POR_ID.get(int(modulo))
    else:
        caso = CASOS.get(modulo)
    if caso is None:
        raise KeyError(f"Módulo desconocido: {modulo}")
    return caso


def describir() -> list[dict]:
    """Metadatos de los módulos (planteamiento de la semana 1 + parámetros y escenarios)."""
    salida = []
    for base in MODULOS:
        caso = POR_ID[base["id"]]
        salida.append({
            **base,
            "parametros": caso.PARAMETROS,
            "escenarios_definidos": base["escenarios"],
            "escenarios": {t: {"descripcion": e["descripcion"], "parametros": e["parametros"], "fuente": e["fuente"]} for t, e in caso.ESCENARIOS.items()},
            "estado": "Implementado: 3 escenarios",
        })
    return salida


def simular(modulo, tipo: str, parametros: dict | None = None) -> dict:
    """Valida, ejecuta y devuelve el resultado canónico (JSON estricto) de un módulo y escenario."""
    caso = obtener(modulo)
    if tipo not in comun.TIPOS:
        raise comun.ErrorParametros([f"Escenario desconocido: {tipo}"])
    limpio = comun.validar(caso.PARAMETROS, parametros, tipo, caso.ESCENARIOS[tipo]["parametros"])
    resultado, ms = comun.cronometrar(lambda: caso.ejecutar(tipo, limpio))
    resultado.update({
        "modulo": caso.CODIGO,
        "modulo_id": caso.ID,
        "nombre_modulo": caso.NOMBRE,
        "tipo": tipo,
        "parametros": limpio,
        "fuente_datos": caso.ESCENARIOS[tipo]["fuente"],
        "modelo_version": VERSION_MODELO,
        "es_subexperimento": tipo == "MAL_CONDICIONADO",
        "tiempo_total_ms": round(ms, 2),
    })
    return comun.limpiar(resultado)
