"""Teoría de errores transversal a todas las familias de métodos."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

REFERENCIA_NULA = 1e-300


def error_absoluto(referencia: float, aproximado: float) -> float:
    """E_abs = |x_ref - x_aprox|."""
    return abs(referencia - aproximado)


def error_relativo(referencia: float, aproximado: float) -> float | None:
    """E_rel = |x_ref - x_aprox| / |x_ref|; None si la referencia es nula."""
    if abs(referencia) < REFERENCIA_NULA:
        return None
    return abs(referencia - aproximado) / abs(referencia)


def error_porcentual(referencia: float, aproximado: float) -> float | None:
    """E_% = 100 E_rel; None si la referencia es nula."""
    relativo = error_relativo(referencia, aproximado)
    return None if relativo is None else 100.0 * relativo


def error_absoluto_vector(referencia: Sequence[float], aproximado: Sequence[float]) -> float:
    """Norma euclídea del error ||x_ref - x_aprox||_2."""
    return float(np.linalg.norm(np.asarray(referencia, float) - np.asarray(aproximado, float)))


def error_relativo_vector(referencia: Sequence[float], aproximado: Sequence[float]) -> float | None:
    """||x_ref - x_aprox||_2 / ||x_ref||_2; None si la referencia es nula."""
    norma = float(np.linalg.norm(np.asarray(referencia, float)))
    if norma < REFERENCIA_NULA:
        return None
    return error_absoluto_vector(referencia, aproximado) / norma


def residual_lineal(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Residuo relativo ||b - A x||_2 / ||b||_2; absoluto si b es nulo."""
    r = float(np.linalg.norm(b - A @ x))
    norma_b = float(np.linalg.norm(b))
    return r / norma_b if norma_b > REFERENCIA_NULA else r


def numero_condicion_relativo(
    entrada: float, entrada_perturbada: float, salida: float, salida_perturbada: float
) -> float | None:
    """Sensibilidad estimada kappa = |dS/S| / |dE/E| por perturbación finita."""
    if abs(entrada) < REFERENCIA_NULA or abs(salida) < REFERENCIA_NULA or entrada == entrada_perturbada:
        return None
    return abs((salida_perturbada - salida) / salida) / abs((entrada_perturbada - entrada) / entrada)


def orden_observado(error_grueso: float, error_fino: float, razon_paso: float = 2.0) -> float | None:
    """p = log(e_h / e_{h/r}) / log(r); None si algún error no es positivo."""
    if error_grueso is None or error_fino is None or error_grueso <= 0 or error_fino <= 0:
        return None
    return math.log(error_grueso / error_fino) / math.log(razon_paso)
