"""Interpolación: Lagrange, Newton por diferencias divididas y spline cúbico natural.

Con los mismos nodos, Lagrange y Newton construyen el MISMO polinomio de grado <= n; sólo
difieren en el redondeo. El spline cúbico natural resuelve un sistema tridiagonal SPD para
los momentos M_i = s''(t_i), que se resuelve con los métodos de sistemas lineales de la semana 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import sistemas


def lagrange(nodos: Sequence[float], valores: Sequence[float], t) -> np.ndarray:
    """p(t) = Σ y_i L_i(t), L_i(t) = Π_{j≠i} (t − t_j)/(t_i − t_j). Forma directa O(n²) por punto."""
    x = np.asarray(nodos, float)
    y = np.asarray(valores, float)
    t = np.atleast_1d(np.asarray(t, float))
    resultado = np.zeros_like(t)
    for i in range(len(x)):
        base = np.ones_like(t)
        for j in range(len(x)):
            if j != i:
                base *= (t - x[j]) / (x[i] - x[j])
        resultado += y[i] * base
    return resultado


def diferencias_divididas(nodos: Sequence[float], valores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (coeficientes, tabla). coef[k] = f[t_0, ..., t_k]; tabla[i, k] = f[t_i, ..., t_{i+k}]."""
    x = np.asarray(nodos, float)
    n = len(x)
    tabla = np.full((n, n), np.nan)
    tabla[:, 0] = np.asarray(valores, float)
    for k in range(1, n):
        for i in range(n - k):
            tabla[i, k] = (tabla[i + 1, k - 1] - tabla[i, k - 1]) / (x[i + k] - x[i])
    return tabla[0, :].copy(), tabla


def newton_evaluar(nodos: Sequence[float], coeficientes: Sequence[float], t) -> np.ndarray:
    """Evaluación de Horner: p(t) = c_0 + (t − t_0)(c_1 + (t − t_1)(c_2 + ...))."""
    x = np.asarray(nodos, float)
    t = np.atleast_1d(np.asarray(t, float))
    p = np.full_like(t, coeficientes[-1])
    for k in range(len(coeficientes) - 2, -1, -1):
        p = p * (t - x[k]) + coeficientes[k]
    return p


def newton_derivada(nodos: Sequence[float], coeficientes: Sequence[float], t) -> np.ndarray:
    """Derivada del polinomio de Newton por Horner extendido."""
    x = np.asarray(nodos, float)
    t = np.atleast_1d(np.asarray(t, float))
    p = np.full_like(t, coeficientes[-1])
    dp = np.zeros_like(t)
    for k in range(len(coeficientes) - 2, -1, -1):
        dp = dp * (t - x[k]) + p
        p = p * (t - x[k]) + coeficientes[k]
    return dp


@dataclass
class SplineCubico:
    nodos: np.ndarray
    valores: np.ndarray
    momentos: np.ndarray  # M_i = s''(t_i)
    metodo_sistema: str
    iteraciones_sistema: int
    residuo_sistema: float
    kappa_sistema: float

    def _tramo(self, t: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.nodos, t, side="right") - 1, 0, len(self.nodos) - 2)

    def __call__(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        i = self._tramo(t)
        x, y, M = self.nodos, self.valores, self.momentos
        h = x[i + 1] - x[i]
        a, b = x[i + 1] - t, t - x[i]
        return (M[i] * a**3 + M[i + 1] * b**3) / (6 * h) + (y[i] / h - M[i] * h / 6) * a + (y[i + 1] / h - M[i + 1] * h / 6) * b

    def derivada(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        i = self._tramo(t)
        x, y, M = self.nodos, self.valores, self.momentos
        h = x[i + 1] - x[i]
        a, b = x[i + 1] - t, t - x[i]
        return (-M[i] * a**2 + M[i + 1] * b**2) / (2 * h) - (y[i] / h - M[i] * h / 6) + (y[i + 1] / h - M[i + 1] * h / 6)


def sistema_spline_natural(nodos: Sequence[float], valores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Sistema para los momentos interiores M_1..M_{n−1} con M_0 = M_n = 0 (spline natural).

    h_{i−1} M_{i−1} + 2(h_{i−1} + h_i) M_i + h_i M_{i+1} = 6 [(y_{i+1} − y_i)/h_i − (y_i − y_{i−1})/h_{i−1}].
    La matriz es tridiagonal, simétrica y estrictamente diagonal dominante con diagonal positiva: SPD.
    """
    x = np.asarray(nodos, float)
    y = np.asarray(valores, float)
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("Los nodos deben ser estrictamente crecientes.")
    m = len(x) - 2
    A = np.zeros((m, m))
    b = np.zeros(m)
    for k in range(m):
        i = k + 1
        A[k, k] = 2.0 * (h[i - 1] + h[i])
        if k > 0:
            A[k, k - 1] = h[i - 1]
        if k < m - 1:
            A[k, k + 1] = h[i]
        b[k] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return A, b


def spline_natural(nodos: Sequence[float], valores: Sequence[float], metodo: str = "Gradiente conjugado", tol: float = 1e-13) -> SplineCubico:
    x = np.asarray(nodos, float)
    y = np.asarray(valores, float)
    if len(x) < 3:
        raise ValueError("El spline natural requiere al menos 3 nodos.")
    A, b = sistema_spline_natural(x, y)
    if not np.any(b):
        interiores = np.zeros(len(b))
        iteraciones, residuo = 0, 0.0
    elif metodo == "Gradiente conjugado":
        r = sistemas.gradiente_conjugado(A, b, tol=tol)
        interiores, iteraciones, residuo = r.x, r.iteraciones, r.historial_residual[-1] if r.historial_residual else 0.0
    elif metodo == "Gauss-Seidel":
        r = sistemas.gauss_seidel(A, b, tol=tol, max_iter=5000)
        interiores, iteraciones, residuo = r.x, r.iteraciones, r.historial_residual[-1]
    else:
        r = sistemas.resolver_lu(A, b)
        interiores, iteraciones, residuo = r.x, 1, r.historial_residual[-1]
    momentos = np.concatenate([[0.0], interiores, [0.0]])
    return SplineCubico(x, y, momentos, metodo, iteraciones, float(residuo), sistemas.numero_condicion(A))


def constante_lebesgue(nodos: Sequence[float], malla) -> float:
    """Λ = max_t Σ |L_i(t)|: cota de amplificación del ruido de los datos en la interpolación polinómica."""
    x = np.asarray(nodos, float)
    malla = np.asarray(malla, float)
    suma = np.zeros_like(malla)
    for i in range(len(x)):
        base = np.ones_like(malla)
        for j in range(len(x)):
            if j != i:
                base *= (malla - x[j]) / (x[i] - x[j])
        suma += np.abs(base)
    return float(suma.max())
