"""Solución de sistemas lineales A x = b.

Condiciones de convergencia usadas en los experimentos:
    - Jacobi converge si y sólo si rho(T_J) < 1, T_J = I - D^-1 A. Con A SPD equivale
      a que 2D - A también sea SPD; por tanto una matriz SPD no garantiza Jacobi.
    - Gauss-Seidel converge para toda A SPD.
    - Gradiente conjugado exige A SPD; en aritmética exacta termina en a lo sumo n pasos.

El criterio de parada común de los iterativos es ||b - A x||_2 / ||b||_2 <= tol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UMBRAL_DIVERGENCIA = 1e100


@dataclass
class ResultadoSistema:
    x: np.ndarray
    iteraciones: int
    convergio: bool
    criterio: str
    historial_x: list[np.ndarray] = field(default_factory=list)
    historial_residual: list[float] = field(default_factory=list)


def _residuo_relativo(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))


def es_spd(A: np.ndarray) -> bool:
    """Comprueba simetría y definición positiva mediante Cholesky."""
    if A.shape[0] != A.shape[1] or not np.allclose(A, A.T, rtol=1e-12, atol=1e-14):
        return False
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return False
    return True


def numero_condicion(A: np.ndarray) -> float:
    """Número de condición espectral kappa_2(A) = sigma_max / sigma_min."""
    return float(np.linalg.cond(A, 2))


def radio_espectral_jacobi(A: np.ndarray) -> float:
    """rho(I - D^-1 A)."""
    D_inv = np.diag(1.0 / np.diag(A))
    return float(np.max(np.abs(np.linalg.eigvals(np.eye(A.shape[0]) - D_inv @ A))))


def radio_espectral_gauss_seidel(A: np.ndarray) -> float:
    """rho(-(D + L)^-1 U), con A = L + D + U."""
    T = -np.linalg.solve(np.tril(A), np.triu(A, 1))
    return float(np.max(np.abs(np.linalg.eigvals(T))))


def _iterar(A, b, x0, tol, max_iter, barrido) -> ResultadoSistema:
    x = np.array(x0, dtype=float)
    historial_x, historial_r = [], []
    for k in range(1, max_iter + 1):
        x = barrido(x)
        norma = float(np.linalg.norm(x))
        if not np.isfinite(norma) or norma > UMBRAL_DIVERGENCIA:
            return ResultadoSistema(x, k, False, "divergencia: ||x|| excede 1e100", historial_x, historial_r)
        residuo = _residuo_relativo(A, x, b)
        historial_x.append(x.copy())
        historial_r.append(residuo)
        if residuo <= tol:
            return ResultadoSistema(x, k, True, f"||r||/||b|| <= {tol:g}", historial_x, historial_r)
    return ResultadoSistema(x, max_iter, False, f"se alcanzó max_iter = {max_iter}", historial_x, historial_r)


def jacobi(A: np.ndarray, b: np.ndarray, tol: float = 1e-10, max_iter: int = 1000, x0=None) -> ResultadoSistema:
    """x^(k+1) = D^-1 (b - (L + U) x^(k))."""
    d = np.diag(A)
    resto = A - np.diag(d)
    x_inicial = np.zeros_like(b, dtype=float) if x0 is None else x0
    return _iterar(A, b, x_inicial, tol, max_iter, lambda x: (b - resto @ x) / d)


def gauss_seidel(A: np.ndarray, b: np.ndarray, tol: float = 1e-10, max_iter: int = 1000, x0=None) -> ResultadoSistema:
    """x_i^(k+1) = (b_i - sum_{j<i} a_ij x_j^(k+1) - sum_{j>i} a_ij x_j^(k)) / a_ii."""
    n = A.shape[0]

    def barrido(x):
        nuevo = x.copy()
        for i in range(n):
            nuevo[i] = (b[i] - A[i, :i] @ nuevo[:i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]
        return nuevo

    x_inicial = np.zeros_like(b, dtype=float) if x0 is None else x0
    return _iterar(A, b, x_inicial, tol, max_iter, barrido)


def factorizar_lu(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factorización P A = L U por eliminación con pivoteo parcial (L con diagonal unitaria)."""
    n = A.shape[0]
    U = np.array(A, dtype=float)
    L = np.eye(n)
    P = np.arange(n)
    for j in range(n - 1):
        pivote = j + int(np.argmax(np.abs(U[j:, j])))
        if U[pivote, j] == 0.0:
            raise np.linalg.LinAlgError("Matriz singular en la factorización LU.")
        if pivote != j:
            U[[j, pivote], :] = U[[pivote, j], :]
            L[[j, pivote], :j] = L[[pivote, j], :j]
            P[[j, pivote]] = P[[pivote, j]]
        L[j + 1 :, j] = U[j + 1 :, j] / U[j, j]
        U[j + 1 :, j:] -= np.outer(L[j + 1 :, j], U[j, j:])
    if U[n - 1, n - 1] == 0.0:
        raise np.linalg.LinAlgError("Matriz singular en la factorización LU.")
    return L, U, P


def sustitucion_adelante(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resuelve L z = y con L triangular inferior."""
    z = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        z[i] = (y[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def sustitucion_atras(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resuelve U x = y con U triangular superior."""
    n = len(y)
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


def resolver_lu(A: np.ndarray, b: np.ndarray) -> ResultadoSistema:
    """Método directo: L z = P b y U x = z."""
    L, U, P = factorizar_lu(A)
    x = sustitucion_atras(U, sustitucion_adelante(L, b[P]))
    return ResultadoSistema(x, 1, True, "método directo: P A = L U", [x.copy()], [_residuo_relativo(A, x, b)])


def gradiente_conjugado(
    A: np.ndarray, b: np.ndarray, tol: float = 1e-10, max_iter: int | None = None, x0=None
) -> ResultadoSistema:
    """Gradiente conjugado estándar para A SPD. El residuo reportado es el verdadero, b - A x."""
    n = A.shape[0]
    limite = max_iter if max_iter is not None else 2 * n
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    r = b - A @ x
    p = r.copy()
    rr = float(r @ r)
    historial_x, historial_r = [], []
    for k in range(1, limite + 1):
        Ap = A @ p
        curvatura = float(p @ Ap)
        if curvatura <= 0.0:
            return ResultadoSistema(x, k - 1, False, "ruptura: p^T A p <= 0 por redondeo", historial_x, historial_r)
        alfa = rr / curvatura
        x = x + alfa * p
        r = r - alfa * Ap
        rr_nuevo = float(r @ r)
        residuo = _residuo_relativo(A, x, b)
        historial_x.append(x.copy())
        historial_r.append(residuo)
        if residuo <= tol:
            return ResultadoSistema(x, k, True, f"||r||/||b|| <= {tol:g}", historial_x, historial_r)
        p = r + (rr_nuevo / rr) * p
        rr = rr_nuevo
    return ResultadoSistema(x, limite, False, f"se alcanzó max_iter = {limite}", historial_x, historial_r)


def matriz_tramos_cable(n: int, acoplamiento: float) -> np.ndarray:
    """Matriz A = I + lambda L de calibración por tramos de cable.

    L es el laplaciano de un camino de n tramos: penaliza diferencias de desgaste
    entre tramos vecinos. Es SPD para lambda >= 0, con autovalores en [1, 1 + 4 lambda).
    """
    laplaciano = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    laplaciano[0, 0] = laplaciano[n - 1, n - 1] = 1.0
    return np.eye(n) + acoplamiento * laplaciano


def matriz_hilbert(n: int) -> np.ndarray:
    """Matriz de Gram de la base monomial en [0, 1]: H_ij = 1 / (i + j + 1). Es SPD."""
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
