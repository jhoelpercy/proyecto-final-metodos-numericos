"""Integración numérica: trapecio compuesto, Simpson 1/3 compuesto y Romberg.

Hay dos variantes de cada regla:
    - sobre una función f(t) evaluable en cualquier punto (tablas de refinamiento);
    - sobre muestras equiespaciadas y_0..y_N con paso h (por ejemplo, la salida de RK4).

Errores teóricos: trapecio O(h²), Simpson O(h⁴), Romberg de nivel j O(h^(2j+2)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Funcion = Callable[[float], float]


def trapecio(f: Funcion, a: float, b: float, n: int) -> float:
    """T(h) = h [f(a)/2 + Σ f(a + i h) + f(b)/2], h = (b − a)/n."""
    if n < 1:
        raise ValueError("El trapecio requiere n >= 1 subintervalos.")
    t = np.linspace(a, b, n + 1)
    y = np.array([f(float(ti)) for ti in t])
    return trapecio_datos(y, (b - a) / n)


def simpson(f: Funcion, a: float, b: float, n: int) -> float:
    """S(h) = h/3 [f_0 + 4 Σ f_impar + 2 Σ f_par + f_n]; n debe ser par."""
    if n < 2 or n % 2:
        raise ValueError("Simpson 1/3 requiere un número par de subintervalos.")
    t = np.linspace(a, b, n + 1)
    y = np.array([f(float(ti)) for ti in t])
    return simpson_datos(y, (b - a) / n)


def trapecio_datos(y: Sequence[float], h: float) -> float:
    y = np.asarray(y, float)
    return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))


def simpson_datos(y: Sequence[float], h: float) -> float:
    """Simpson 1/3 compuesto; si el número de subintervalos es impar, el último se integra con Simpson 3/8."""
    y = np.asarray(y, float)
    n = len(y) - 1
    if n < 2:
        raise ValueError("Simpson requiere al menos 2 subintervalos.")
    if n % 2 == 0:
        return float(h / 3.0 * (y[0] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum() + y[-1]))
    # n impar: Simpson 1/3 en los primeros n−3 y 3/8 en los 3 últimos (mismo orden h⁴).
    cabeza = simpson_datos(y[: n - 2], h) if n - 3 >= 2 else 0.0
    cola = 3.0 * h / 8.0 * (y[-4] + 3.0 * y[-3] + 3.0 * y[-2] + y[-1])
    return float(cabeza + cola)


@dataclass
class ResultadoRomberg:
    valor: float
    niveles: int
    convergio: bool
    criterio: str
    tabla: list[list[float]] = field(default_factory=list)  # R[i][j], i = refinamiento, j = extrapolación
    evaluaciones: int = 0


def romberg(f: Funcion, a: float, b: float, tol: float = 1e-10, max_niveles: int = 12) -> ResultadoRomberg:
    """R(i,0) = trapecio con 2^i subintervalos; R(i,j) = R(i,j−1) + (R(i,j−1) − R(i−1,j−1))/(4^j − 1).

    Reutiliza las evaluaciones anteriores: cada nivel sólo evalúa los puntos medios nuevos.
    Parada: |R(i,i) − R(i−1,i−1)| <= tol.
    """
    h = b - a
    tabla = [[0.5 * h * (f(a) + f(b))]]
    evaluaciones = 2
    for i in range(1, max_niveles + 1):
        h /= 2.0
        nuevos = sum(f(a + (2 * k - 1) * h) for k in range(1, 2 ** (i - 1) + 1))
        evaluaciones += 2 ** (i - 1)
        fila = [0.5 * tabla[-1][0] + h * nuevos]
        for j in range(1, i + 1):
            fila.append(fila[j - 1] + (fila[j - 1] - tabla[-1][j - 1]) / (4.0**j - 1.0))
        tabla.append(fila)
        if abs(fila[-1] - tabla[-2][-1]) <= tol:
            return ResultadoRomberg(fila[-1], i, True, f"|R(i,i) − R(i−1,i−1)| <= {tol:g}", tabla, evaluaciones)
    return ResultadoRomberg(tabla[-1][-1], max_niveles, False, f"se alcanzó max_niveles = {max_niveles}", tabla, evaluaciones)


def romberg_datos(y: Sequence[float], h: float) -> ResultadoRomberg:
    """Romberg con muestras equiespaciadas: usa los trapecios con pasos h·2^j que caben en la malla.

    Con N subintervalos, el número de niveles es 1 + (mayor j tal que 2^j divide a N).
    """
    y = np.asarray(y, float)
    n = len(y) - 1
    saltos = 1
    while n % (saltos * 2) == 0 and n // (saltos * 2) >= 1:
        saltos *= 2
    # Trapecios desde el paso más grueso (h·saltos) al más fino (h).
    trapecios = []
    s = saltos
    while s >= 1:
        trapecios.append(trapecio_datos(y[::s], h * s))
        s //= 2
    tabla = []
    for i, t in enumerate(trapecios):
        fila = [t]
        for j in range(1, i + 1):
            fila.append(fila[j - 1] + (fila[j - 1] - tabla[i - 1][j - 1]) / (4.0**j - 1.0))
        tabla.append(fila)
    niveles = len(tabla) - 1
    return ResultadoRomberg(tabla[-1][-1], niveles, True, f"{niveles + 1} trapecios con pasos h·2^j de la malla", tabla, n + 1)
