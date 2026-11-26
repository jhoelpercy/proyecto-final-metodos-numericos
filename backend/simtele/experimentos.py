"""Experimentos reproducibles: los 12 escenarios (4 módulos × 3 condiciones).

Uso, desde la raíz del repositorio:
    python -m backend.simtele.experimentos

Imprime un resumen por módulo y escenario y escribe frontend/datos/experimentos.json, que la
aplicación usa para el panel de comparación y como respaldo estático cuando no hay API.
generar() conserva el experimento de sistemas lineales de la semana 1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import VERSION_MODELO
from .modulos import MODULOS, NOTA_SINTETICOS
from .numerico import errores, sistemas

SALIDA = Path(__file__).resolve().parents[2] / "frontend" / "datos" / "experimentos.json"
PANEL = SALIDA.with_name("panel.json")
MAX_ITERACIONES_GRAFICO = 400

ESCENARIOS = {
    "NORMAL": {"descripcion": "A = I + 0.1 L con 6 tramos", "tol": 1e-10, "max_iter": 500},
    "ESTRES": {"descripcion": "A = I + 5 L con 40 tramos", "tol": 1e-12, "max_iter": 5000},
    "MAL_CONDICIONADO": {"descripcion": "Hilbert de orden 8", "tol": 1e-10, "max_iter": 5000},
}


def construir_sistema(tipo: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (A, x*, b) con A SPD y b = A x*. x* está en % de desgaste por tramo."""
    if tipo == "MAL_CONDICIONADO":
        A = sistemas.matriz_hilbert(8)
        x_exacta = np.ones(8)
    else:
        n, lam = (6, 0.1) if tipo == "NORMAL" else (40, 5.0)
        A = sistemas.matriz_tramos_cable(n, lam)
        x_exacta = 20.0 + 10.0 * np.sin(2.0 * np.pi * np.arange(n) / n)
    return A, x_exacta, A @ x_exacta


def _finito(valor: float) -> float | None:
    valor = float(valor)
    return valor if np.isfinite(valor) else None


def _submuestrear(valores: list[float]) -> list[list[float]]:
    salto = max(1, len(valores) // MAX_ITERACIONES_GRAFICO)
    return [[i + 1, valores[i]] for i in range(0, len(valores), salto)]


def amplificacion_perturbacion(A: np.ndarray, b: np.ndarray, relativa: float = 1e-8) -> float:
    """(||dx||/||x||) / (||db||/||b||) con db en la dirección singular más desfavorable.

    La cota superior teórica de esta razón es kappa_2(A).
    """
    U, _, _ = np.linalg.svd(A)
    delta = relativa * np.linalg.norm(b) * U[:, -1]
    x_base = sistemas.resolver_lu(A, b).x
    x_perturbada = sistemas.resolver_lu(A, b + delta).x
    return float(np.linalg.norm(x_perturbada - x_base) / np.linalg.norm(x_base) / relativa)


def correr_escenario(tipo: str) -> dict:
    """Ejecuta los cuatro métodos y la referencia directa, con la misma tolerancia."""
    configuracion = ESCENARIOS[tipo]
    A, x_exacta, b = construir_sistema(tipo)
    tol, max_iter = configuracion["tol"], configuracion["max_iter"]
    corridas = [
        ("Jacobi", lambda: sistemas.jacobi(A, b, tol, max_iter)),
        ("Gauss-Seidel", lambda: sistemas.gauss_seidel(A, b, tol, max_iter)),
        ("LU con pivoteo parcial", lambda: sistemas.resolver_lu(A, b)),
        ("Gradiente conjugado", lambda: sistemas.gradiente_conjugado(A, b, tol, max_iter)),
    ]
    metodos = []
    for nombre, correr in corridas:
        inicio = time.perf_counter()
        r = correr()
        tiempo_ms = (time.perf_counter() - inicio) * 1000.0
        finito = bool(np.all(np.isfinite(r.x)))
        metodos.append(
            {
                "metodo": nombre,
                "iteraciones": r.iteraciones,
                "convergio": r.convergio,
                "criterio": r.criterio,
                "tiempo_ms": round(tiempo_ms, 3),
                "error_abs": _finito(errores.error_absoluto_vector(x_exacta, r.x)) if finito else None,
                "error_rel": _finito(errores.error_relativo_vector(x_exacta, r.x)) if finito else None,
                "residual": _finito(errores.residual_lineal(A, r.x, b)) if finito else None,
                "historial_residual": _submuestrear([v if np.isfinite(v) else None for v in r.historial_residual]),
            }
        )
    referencia = np.linalg.solve(A, b)
    return {
        "tipo": tipo,
        "descripcion": configuracion["descripcion"],
        "tolerancia": tol,
        "max_iter": max_iter,
        "n": len(b),
        "kappa": sistemas.numero_condicion(A),
        "rho_jacobi": sistemas.radio_espectral_jacobi(A),
        "rho_gauss_seidel": sistemas.radio_espectral_gauss_seidel(A),
        "es_spd": sistemas.es_spd(A),
        "error_referencia_directa": errores.error_absoluto_vector(x_exacta, referencia),
        "amplificacion_perturbacion": amplificacion_perturbacion(A, b),
        "metodos": metodos,
    }


def generar() -> dict:
    return {
        "modelo_version": VERSION_MODELO,
        "nota": NOTA_SINTETICOS,
        "modulos": MODULOS,
        "sistemas": [correr_escenario(t) for t in ESCENARIOS],
    }


def imprimir(datos: dict) -> None:
    for e in datos["sistemas"]:
        print(f"\n{e['tipo']}: {e['descripcion']}; n = {e['n']}, κ₂ = {e['kappa']:.3e}, ρ(J) = {e['rho_jacobi']:.4f}, ρ(GS) = {e['rho_gauss_seidel']:.4f}, SPD = {e['es_spd']}, amplificación δb→δx = {e['amplificacion_perturbacion']:.3e}")
        print(f"{'Método':<24} {'Iter.':>6} {'Conv.':>6} {'||x-x*||':>11} {'Rel.':>11} {'Residuo':>11}")
        for m in e["metodos"]:
            fmt = lambda v: "—" if v is None else f"{v:.3e}"
            print(f"{m['metodo']:<24} {m['iteraciones']:>6} {'Sí' if m['convergio'] else 'No':>6} {fmt(m['error_abs']):>11} {fmt(m['error_rel']):>11} {fmt(m['residual']):>11}")


def generar_todo() -> dict:
    """Ejecuta los 12 escenarios con sus parámetros por defecto."""
    from .casos import CASOS, describir, simular

    datos = generar()
    datos["modulos"] = describir()
    datos["resultados"] = {codigo: {tipo: simular(codigo, tipo) for tipo in caso.ESCENARIOS} for codigo, caso in CASOS.items()}
    return datos


def imprimir_casos(datos: dict) -> None:
    print("\nResumen de los 12 escenarios")
    for codigo, por_tipo in datos["resultados"].items():
        for tipo, r in por_tipo.items():
            panel = "; ".join(f"{k}: {v}" for k, v in r["panel"])
            print(f"  M{r['modulo_id']} {tipo:<17} {r['tiempo_total_ms']:8.0f} ms  {panel}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    datos = generar_todo()
    imprimir(datos)
    imprimir_casos(datos)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    panel = {
        "modelo_version": datos["modelo_version"],
        "panel": {
            codigo: {
                tipo: {
                    "panel": r["panel"],
                    "conclusiones": r["conclusiones"],
                    "tiempo_total_ms": r["tiempo_total_ms"],
                    "convergencias": [[e["familia"], e["etiqueta"], e["convergio"]] for e in r["ejecuciones"]],
                }
                for tipo, r in por_tipo.items()
            }
            for codigo, por_tipo in datos["resultados"].items()
        },
    }
    PANEL.write_text(json.dumps(panel, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    print(f"\nDatos de la aplicación: {SALIDA} y {PANEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
