"""Ecuaciones diferenciales ordinarias y' = f(t, y) con paso fijo.

Variante de RK2 declarada: método de Heun (trapecio explícito),
    k1 = f(t_n, y_n),  k2 = f(t_n + h, y_n + h k1),  y_{n+1} = y_n + h (k1 + k2)/2.
RK4 clásico:
    y_{n+1} = y_n + h (k1 + 2 k2 + 2 k3 + k4)/6.
Errores globales teóricos: Heun O(h²), RK4 O(h⁴).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Campo = Callable[[float, np.ndarray], np.ndarray]


def paso_heun(f: Campo, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + 0.5 * h * (k1 + k2)


def paso_rk4(f: Campo, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


PASOS = {"Heun (RK2)": paso_heun, "RK4": paso_rk4}


@dataclass
class ResultadoEDO:
    t: np.ndarray
    y: np.ndarray  # forma (len(t), dimensión)
    pasos: int
    estable: bool
    criterio: str


def integrar(f: Campo, y0, t0: float, t_fin: float, h: float, metodo: str = "RK4", limite: float = 1e12) -> ResultadoEDO:
    """Integra con paso fijo h de t0 a t_fin (el último paso se recorta para llegar exacto a t_fin)."""
    paso = PASOS[metodo]
    n = int(np.ceil((t_fin - t0) / h - 1e-12))
    t = np.empty(n + 1)
    y = np.empty((n + 1, len(np.atleast_1d(y0))))
    t[0], y[0] = t0, np.atleast_1d(np.asarray(y0, float))
    for i in range(n):
        hi = min(h, t_fin - t[i])
        y[i + 1] = paso(f, t[i], y[i], hi)
        t[i + 1] = t[i] + hi
        if not np.all(np.isfinite(y[i + 1])) or np.max(np.abs(y[i + 1])) > limite:
            return ResultadoEDO(t[: i + 2], y[: i + 2], i + 1, False, f"solución no acotada (|y| > {limite:g}) en t = {t[i + 1]:.4g}")
    return ResultadoEDO(t, y, n, True, f"t = {t_fin:g} alcanzado con {n} pasos")


def integrar_hasta_evento(
    f: Campo,
    y0,
    h: float,
    evento: Callable[[np.ndarray], float],
    metodo: str = "RK4",
    t_max: float = 1e4,
    buscador: Callable | None = None,
):
    """Integra con paso fijo hasta que evento(y) cambia de signo y ajusta el último paso.

    El último paso h* se obtiene resolviendo g(h*) = evento(Φ_h*(y_n)) = 0, donde Φ es un paso del
    mismo método. buscador(g, 0, h) debe devolver un objeto con atributos raiz, iteraciones e historial
    (por ejemplo raices.secante o raices.biseccion envueltos). Devuelve (ResultadoEDO, resultado_raiz).
    """
    paso = PASOS[metodo]
    t, y = [0.0], [np.atleast_1d(np.asarray(y0, float))]
    while t[-1] < t_max:
        siguiente = paso(f, t[-1], y[-1], h)
        if evento(siguiente) <= 0.0:
            base_t, base_y = t[-1], y[-1]

            def g(hs: float) -> float:
                return evento(paso(f, base_t, base_y, hs))

            raiz = buscador(g, 0.0, h)
            h_final = raiz.raiz
            y.append(paso(f, base_t, base_y, h_final))
            t.append(base_t + h_final)
            return ResultadoEDO(np.array(t), np.array(y), len(t) - 1, True, "evento v = 0 localizado en el último paso"), raiz
        t.append(t[-1] + h)
        y.append(siguiente)
        if not np.all(np.isfinite(siguiente)):
            break
    return ResultadoEDO(np.array(t), np.array(y), len(t) - 1, False, "no se alcanzó el evento"), None
