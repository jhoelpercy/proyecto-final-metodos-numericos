"""Raíces de ecuaciones no lineales f(x) = 0: bisección, Newton-Raphson y secante.

Cada método devuelve la tabla completa de iteraciones para poder guardarla y graficarla.
Criterios de parada:
    - Bisección: semiancho del intervalo (b - a)/2 <= tol, o f(c) = 0 exacto.
    - Newton y secante: |x_k - x_{k-1}| <= tol, o f(x_k) = 0 exacto.
Todos tienen tope de iteraciones y reportan el motivo de parada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

Funcion = Callable[[float], float]


@dataclass
class ResultadoRaiz:
    raiz: float
    iteraciones: int
    convergio: bool
    criterio: str
    # Cada fila: {"k", "x", "fx", "paso"} (bisección agrega "a" y "b").
    historial: list[dict] = field(default_factory=list)


def biseccion(f: Funcion, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> ResultadoRaiz:
    """Bisección sobre [a, b]; exige cambio de signo f(a) f(b) < 0."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return ResultadoRaiz(a, 0, True, "f(a) = 0 exacto", [])
    if fb == 0.0:
        return ResultadoRaiz(b, 0, True, "f(b) = 0 exacto", [])
    if fa * fb > 0:
        raise ValueError("Bisección requiere f(a) y f(b) de signo opuesto.")
    historial = []
    c = a
    for k in range(1, max_iter + 1):
        c = a + (b - a) / 2.0
        fc = f(c)
        semiancho = (b - a) / 2.0
        historial.append({"k": k, "a": a, "b": b, "x": c, "fx": fc, "paso": semiancho})
        if fc == 0.0:
            return ResultadoRaiz(c, k, True, "f(c) = 0 exacto", historial)
        if semiancho <= tol:
            return ResultadoRaiz(c, k, True, f"(b − a)/2 <= {tol:g}", historial)
        if math.copysign(1.0, fa) * math.copysign(1.0, fc) < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return ResultadoRaiz(c, max_iter, False, f"se alcanzó max_iter = {max_iter}", historial)


def newton_raphson(f: Funcion, df: Funcion, x0: float, tol: float = 1e-10, max_iter: int = 100) -> ResultadoRaiz:
    """x_{k+1} = x_k − f(x_k)/f'(x_k). Se detiene si la derivada se anula."""
    x = x0
    historial = []
    for k in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if fx == 0.0:
            historial.append({"k": k, "x": x, "fx": fx, "paso": 0.0})
            return ResultadoRaiz(x, k, True, "f(x) = 0 exacto", historial)
        if dfx == 0.0 or not math.isfinite(dfx):
            return ResultadoRaiz(x, k - 1, False, "derivada nula o no finita", historial)
        nuevo = x - fx / dfx
        paso = abs(nuevo - x)
        x = nuevo
        if not math.isfinite(x):
            return ResultadoRaiz(x, k, False, "iterado no finito: divergencia", historial)
        historial.append({"k": k, "x": x, "fx": f(x), "paso": paso})
        if paso <= tol:
            return ResultadoRaiz(x, k, True, f"|x_k − x_(k−1)| <= {tol:g}", historial)
    return ResultadoRaiz(x, max_iter, False, f"se alcanzó max_iter = {max_iter}", historial)


def secante(f: Funcion, x0: float, x1: float, tol: float = 1e-10, max_iter: int = 100) -> ResultadoRaiz:
    """x_{k+1} = x_k − f(x_k)(x_k − x_{k−1})/(f(x_k) − f(x_{k−1}))."""
    f0, f1 = f(x0), f(x1)
    historial = []
    for k in range(1, max_iter + 1):
        if f1 == 0.0:
            return ResultadoRaiz(x1, k - 1, True, "f(x) = 0 exacto", historial)
        denominador = f1 - f0
        if denominador == 0.0:
            return ResultadoRaiz(x1, k - 1, False, "f(x_k) = f(x_(k−1)): secante horizontal", historial)
        x2 = x1 - f1 * (x1 - x0) / denominador
        if not math.isfinite(x2):
            return ResultadoRaiz(x2, k, False, "iterado no finito: divergencia", historial)
        paso = abs(x2 - x1)
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
        historial.append({"k": k, "x": x1, "fx": f1, "paso": paso})
        if paso <= tol:
            return ResultadoRaiz(x1, k, True, f"|x_k − x_(k−1)| <= {tol:g}", historial)
    return ResultadoRaiz(x1, max_iter, False, f"se alcanzó max_iter = {max_iter}", historial)


def orden_convergencia_estimado(iterados: list[float], raiz: float) -> float | None:
    """q ≈ ln(e_{k+1}/e_k) / ln(e_k/e_{k−1}) con los tres últimos errores útiles."""
    errores = [abs(x - raiz) for x in iterados]
    utiles = [e for e in errores if e > 1e-14 * max(1.0, abs(raiz))]
    if len(utiles) < 3:
        return None
    e0, e1, e2 = utiles[-3:]
    if e0 == e1 or e1 == 0 or e0 == 0:
        return None
    denominador = math.log(e1 / e0)
    return None if denominador == 0 else math.log(e2 / e1) / denominador
