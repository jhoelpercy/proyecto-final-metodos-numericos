"""Utilidades comunes de los casos: validación de parámetros, errores, tablas, gráficos y ejecuciones.

Formato canónico del resultado de una simulación (lo que la API devuelve y lo que se guarda):
    {
      "modulo", "modulo_id", "tipo", "parametros", "fuente_datos", "modelo_version",
      "descripcion", "es_subexperimento", "indicadores": [...], "ejecuciones": [...],
      "tablas": [...], "graficos": [...], "conclusiones": {"numerica", "problema", "limites"},
      "panel": [...]
    }
Cada ejecución corresponde a una fila de ejecucion_metodo; sus iteraciones a iteracion_resultado
y su resumen a resultado_resumen.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable

import numpy as np

MAX_FILAS_ITERACION = 400
TIPOS = ("NORMAL", "ESTRES", "MAL_CONDICIONADO")
NOMBRES_TIPO = {"NORMAL": "Normal", "ESTRES": "Estrés", "MAL_CONDICIONADO": "Mal condicionado"}


class ErrorParametros(ValueError):
    """Parámetros fuera de rango; el mensaje lista todos los problemas."""

    def __init__(self, problemas: list[str]):
        super().__init__("; ".join(problemas))
        self.problemas = problemas


def validar(definiciones: list[dict], valores: dict | None, tipo: str, por_defecto: dict) -> dict:
    """Completa con los valores por defecto del escenario y valida rango, tipo y opciones."""
    valores = dict(valores or {})
    limpio, problemas = {}, []
    for d in definiciones:
        clave = d["clave"]
        if "escenarios" in d and tipo not in d["escenarios"]:
            continue
        crudo = valores.get(clave, por_defecto.get(clave))
        if crudo is None:
            problemas.append(f"Falta el parámetro {d['etiqueta']}.")
            continue
        if "opciones" in d:
            if crudo not in d["opciones"]:
                problemas.append(f"{d['etiqueta']}: opción no válida ({crudo}).")
            else:
                limpio[clave] = crudo
            continue
        try:
            numero = float(crudo)
        except (TypeError, ValueError):
            problemas.append(f"{d['etiqueta']}: debe ser numérico.")
            continue
        if not math.isfinite(numero):
            problemas.append(f"{d['etiqueta']}: debe ser finito.")
            continue
        if d.get("entero"):
            if numero != int(numero):
                problemas.append(f"{d['etiqueta']}: debe ser entero.")
                continue
            numero = int(numero)
        if "min" in d and numero < d["min"]:
            problemas.append(f"{d['etiqueta']}: mínimo {d['min']}.")
        elif "max" in d and numero > d["max"]:
            problemas.append(f"{d['etiqueta']}: máximo {d['max']}.")
        limpio[clave] = numero
    if problemas:
        raise ErrorParametros(problemas)
    return limpio


def errores(referencia: float | None, aproximado: float | None) -> tuple[float | None, float | None]:
    """(error absoluto, error relativo); el relativo se omite si |referencia| es despreciable."""
    if referencia is None or aproximado is None or not np.isfinite(aproximado):
        return None, None
    absoluto = abs(float(referencia) - float(aproximado))
    relativo = absoluto / abs(referencia) if abs(referencia) > 1e-12 else None
    return absoluto, relativo


def kappa_relativo(entrada: float, entrada_pert: float, salida: float, salida_pert: float) -> float | None:
    if entrada == 0 or salida == 0 or entrada == entrada_pert:
        return None
    return abs((salida_pert - salida) / salida) / abs((entrada_pert - entrada) / entrada)


def orden(e_grueso: float | None, e_fino: float | None, razon: float = 2.0) -> float | None:
    if not e_grueso or not e_fino or e_grueso <= 0 or e_fino <= 0:
        return None
    # Errores en el piso del redondeo no permiten estimar un orden con sentido.
    if e_fino < 1e-13:
        return None
    return math.log(e_grueso / e_fino) / math.log(razon)


def cronometrar(funcion: Callable[[], Any]) -> tuple[Any, float]:
    inicio = time.perf_counter()
    resultado = funcion()
    return resultado, (time.perf_counter() - inicio) * 1000.0


def submuestrear(filas: list, maximo: int = MAX_FILAS_ITERACION) -> list:
    """Conserva la primera y la última fila y reparte el resto uniformemente."""
    if len(filas) <= maximo:
        return list(filas)
    indices = np.unique(np.linspace(0, len(filas) - 1, maximo).round().astype(int))
    return [filas[i] for i in indices]


def fila(numero: int, aproximacion: dict, residual=None, error_abs=None, error_rel=None) -> dict:
    return {"numero": int(numero), "aproximacion": aproximacion, "residual": residual, "error_abs": error_abs, "error_rel": error_rel}


def ejecucion(
    metodo: str,
    familia: str,
    iteraciones: list[dict],
    *,
    referencia: str,
    valor: dict,
    metricas: dict,
    interpretacion: str,
    convergio: bool = True,
    criterio: str = "",
    tiempo_ms: float = 0.0,
    tolerancia: float | None = None,
    paso: float | None = None,
    max_iter: int | None = None,
    etiqueta: str | None = None,
    total: int | None = None,
) -> dict:
    return {
        "metodo": metodo,
        "etiqueta": etiqueta or metodo,
        "familia": familia,
        "tolerancia": tolerancia,
        "paso": paso,
        "max_iter": max_iter,
        "convergio": bool(convergio),
        "criterio": criterio,
        "tiempo_ms": round(float(tiempo_ms), 4),
        "iteraciones": submuestrear(iteraciones),
        "total_iteraciones": int(total) if total is not None else len(iteraciones),
        "resumen": {"referencia": referencia, "valor": valor, "metricas": metricas, "interpretacion": interpretacion},
    }


def ejecucion_raiz(metodo: str, resultado, raiz_ref: float | None, funcion: Callable[[float], float], *, tol: float, max_iter: int, tiempo_ms: float, contexto: str, unidad: str = "") -> dict:
    """Convierte un ResultadoRaiz en ejecución con error real |x_k − x*| y residuo |f(x_k)|."""
    filas = []
    for h in resultado.historial:
        ea, er = errores(raiz_ref, h["x"])
        aprox = {"x": h["x"], "paso": h["paso"]}
        if "a" in h:
            aprox.update({"a": h["a"], "b": h["b"]})
        filas.append(fila(h["k"], aprox, abs(h["fx"]), ea, er))
    ea, er = errores(raiz_ref, resultado.raiz)
    residuo = abs(funcion(resultado.raiz)) if np.isfinite(resultado.raiz) else None
    return ejecucion(
        metodo,
        "Raíces",
        filas,
        referencia=f"raíz de referencia {fmt(raiz_ref)} {unidad}".strip() if raiz_ref is not None else "sin raíz de referencia",
        valor={"raiz": resultado.raiz},
        metricas={"error_abs": ea, "error_rel": er, "residual": residuo, "iteraciones": resultado.iteraciones},
        interpretacion=f"{contexto}: {metodo} {'converge' if resultado.convergio else 'no converge'} en {resultado.iteraciones} iteraciones ({resultado.criterio}).",
        convergio=resultado.convergio,
        criterio=resultado.criterio,
        tiempo_ms=tiempo_ms,
        tolerancia=tol,
        max_iter=max_iter,
    )


def tabla(titulo: str, columnas: list[str], filas: Iterable[Iterable], nota: str = "") -> dict:
    return {"titulo": titulo, "columnas": columnas, "filas": [list(f) for f in filas], "nota": nota}


def serie(nombre: str, x, y, *, estilo: str = "linea", color: int | None = None, punteada: bool = False, maximo: int = 700) -> dict:
    """Serie de un gráfico. estilo: linea | puntos | linea-puntos. Omite valores no finitos y ceros en log."""
    pares = [[float(a), float(b)] for a, b in zip(x, y) if b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pares) > maximo:
        pares = submuestrear(pares, maximo)
    return {"nombre": nombre, "puntos": pares, "estilo": estilo, "color": color, "punteada": punteada}


def grafico(titulo: str, eje_x: str, eje_y: str, series: list[dict], *, log_x: bool = False, log_y: bool = False, nota: str = "", tipo: str = "linea") -> dict:
    if log_y:
        for s in series:
            s["puntos"] = [p for p in s["puntos"] if p[1] > 0]
    if log_x:
        for s in series:
            s["puntos"] = [p for p in s["puntos"] if p[0] > 0]
    return {"titulo": titulo, "eje_x": eje_x, "eje_y": eje_y, "series": series, "log_x": log_x, "log_y": log_y, "nota": nota, "tipo": tipo}


def indicador(etiqueta: str, valor, unidad: str = "", nota: str = "") -> dict:
    return {"etiqueta": etiqueta, "valor": valor, "unidad": unidad, "nota": nota}


def fmt(v, cifras: int = 6) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    v = float(v)
    if not np.isfinite(v):
        return "no finito"
    if v == 0:
        return "0"
    a = abs(v)
    if a < 1e-3 or a >= 1e6:
        return f"{v:.3e}"
    return f"{v:.{cifras}g}"


def limpiar(objeto):
    """Convierte numpy a tipos nativos y reemplaza NaN/inf por None para JSON estricto."""
    if isinstance(objeto, dict):
        return {str(k): limpiar(v) for k, v in objeto.items()}
    if isinstance(objeto, (list, tuple)):
        return [limpiar(v) for v in objeto]
    if isinstance(objeto, np.ndarray):
        return limpiar(objeto.tolist())
    if isinstance(objeto, (np.bool_,)):
        return bool(objeto)
    if isinstance(objeto, (np.integer,)):
        return int(objeto)
    if isinstance(objeto, (float, np.floating)):
        valor = float(objeto)
        return valor if math.isfinite(valor) else None
    return objeto
