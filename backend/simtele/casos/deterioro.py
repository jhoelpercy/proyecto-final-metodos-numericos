"""Módulo 2: deterioro mecánico y mantenimiento.

Indicador hipotético D(t) = D0 + α t + β t² [%], umbral de inspección D_u.
    f(t) = D(t) − D_u = 0  → bisección, Newton-Raphson y secante.
    Referencia: raíz de la cuadrática con aritmética decimal de 50 dígitos (fórmula estable).
Calibración por tramos: A x = b con A SPD y b = A x* (Jacobi, Gauss-Seidel, LU, gradiente conjugado).
El escenario MAL_CONDICIONADO combina una curva casi tangente al umbral con la matriz de Hilbert.
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext

import numpy as np

from ..numerico import errores as err
from ..numerico import raices, sistemas
from . import comun as c

CODIGO = "deterioro"
ID = 2
NOMBRE = "Deterioro y mantenimiento"

PARAMETROS = [
    {"clave": "D0", "etiqueta": "Deterioro inicial D₀", "unidad": "%", "min": 0, "max": 99, "paso": "any"},
    {"clave": "alfa", "etiqueta": "Tasa lineal α", "unidad": "%/mes", "min": -10, "max": 10, "paso": "any"},
    {"clave": "beta", "etiqueta": "Aceleración β", "unidad": "%/mes²", "min": -1, "max": 1, "paso": "any"},
    {"clave": "Du", "etiqueta": "Umbral de inspección D_u", "unidad": "%", "min": 1, "max": 100, "paso": "any"},
    {"clave": "tol", "etiqueta": "Tolerancia de raíces", "unidad": "", "min": 1e-15, "max": 1e-2, "paso": "any"},
    {"clave": "retraso", "etiqueta": "Retraso de mantenimiento", "unidad": "meses", "min": 0, "max": 24, "paso": 0.5},
    {"clave": "matriz", "etiqueta": "Matriz de calibración", "unidad": "", "opciones": ["tramos", "hilbert"]},
    {"clave": "n", "etiqueta": "Tamaño del sistema n", "unidad": "", "min": 4, "max": 60, "entero": True, "paso": 1},
    {"clave": "lam", "etiqueta": "Acoplamiento λ (A = I + λL)", "unidad": "", "min": 0, "max": 50, "paso": "any"},
    {"clave": "tol_sis", "etiqueta": "Tolerancia de los iterativos", "unidad": "", "min": 1e-15, "max": 1e-2, "paso": "any"},
]

ESCENARIOS = {
    "NORMAL": {
        "descripcion": "Desgaste moderado que cruza el umbral con pendiente clara; sistema de 6 tramos bien condicionado.",
        "parametros": {"D0": 10.0, "alfa": 0.8, "beta": 0.02, "Du": 40.0, "tol": 1e-10, "retraso": 3.0, "matriz": "tramos", "n": 6, "lam": 0.1, "tol_sis": 1e-10},
        "fuente": "Sintético: coeficientes supuestos; x* prefijada (20 + 10 sin(2πi/n) % por tramo).",
    },
    "ESTRES": {
        "descripcion": "Desgaste acelerado (β alto) que llega al umbral en pocos meses; sistema de 40 tramos con fuerte acoplamiento y tolerancias estrictas.",
        "parametros": {"D0": 25.0, "alfa": 2.0, "beta": 0.15, "Du": 40.0, "tol": 1e-13, "retraso": 3.0, "matriz": "tramos", "n": 40, "lam": 5.0, "tol_sis": 1e-12},
        "fuente": "Sintético: condición exigente dentro del mismo modelo.",
    },
    "MAL_CONDICIONADO": {
        "descripcion": "Subexperimento numérico: curva cóncava casi tangente al umbral (máximo 10⁻⁶ % por encima) y matriz de Hilbert n = 8.",
        "parametros": {"D0": 11.200001, "alfa": 2.4, "beta": -0.05, "Du": 40.0, "tol": 1e-10, "retraso": 3.0, "matriz": "hilbert", "n": 8, "lam": 0.0, "tol_sis": 1e-10},
        "fuente": "Sintético: D(t) = 40 + 10⁻⁶ − 0.05 (t − 24)²; Hilbert como matriz SPD mal condicionada conocida.",
    },
}

HORIZONTE = 1200.0  # meses para la búsqueda incremental


def raiz_referencia(D0: float, alfa: float, beta: float, Du: float) -> float | None:
    """Menor raíz positiva de β t² + α t + (D0 − Du) = 0 con 50 dígitos y fórmula sin cancelación."""
    getcontext().prec = 50
    a, b, cc = Decimal(beta), Decimal(alfa), Decimal(D0) - Decimal(Du)
    if a == 0:
        if b == 0:
            return None
        t = -cc / b
        return float(t) if t >= 0 else None
    disc = b * b - 4 * a * cc
    if disc < 0:
        return None
    raiz_disc = disc.sqrt()
    q = -(b + (raiz_disc if b >= 0 else -raiz_disc)) / 2
    candidatas = [q / a] + ([cc / q] if q != 0 else [])
    positivas = sorted(float(t) for t in candidatas if t >= 0)
    return positivas[0] if positivas else None


def _intervalo(f, paso: float = 1.0):
    a, fa = 0.0, f(0.0)
    t = paso
    while t <= HORIZONTE:
        fb = f(t)
        if fa * fb <= 0:
            return a, t
        a, fa = t, fb
        t += paso
    return None


def _sistema(p: dict):
    n = p["n"]
    if p["matriz"] == "hilbert":
        A = sistemas.matriz_hilbert(n)
        x = np.ones(n)
    else:
        A = sistemas.matriz_tramos_cable(n, p["lam"])
        x = 20.0 + 10.0 * np.sin(2.0 * np.pi * np.arange(n) / n)
    return A, x, A @ x


def _amplificacion(A, b, relativa=1e-8):
    U, _, _ = np.linalg.svd(A)
    delta = relativa * np.linalg.norm(b) * U[:, -1]
    x0 = sistemas.resolver_lu(A, b).x
    x1 = sistemas.resolver_lu(A, b + delta).x
    return float(np.linalg.norm(x1 - x0) / np.linalg.norm(x0) / relativa)


def ejecutar(tipo: str, p: dict) -> dict:
    if p["matriz"] == "hilbert" and p["n"] > 12:
        raise c.ErrorParametros(["Con la matriz de Hilbert use n <= 12 (κ crece como e^(3.5 n))."])
    D0, alfa, beta, Du, tol = p["D0"], p["alfa"], p["beta"], p["Du"], p["tol"]
    D = lambda t: D0 + alfa * t + beta * t * t
    f = lambda t: D(t) - Du
    df = lambda t: alfa + 2 * beta * t
    t_ref = raiz_referencia(D0, alfa, beta, Du)
    ejecuciones, tablas, graficos, indicadores = [], [], [], []

    # 1) Raíces.
    intervalo = _intervalo(f)
    resultados = {}
    if t_ref is not None and intervalo is not None:
        a, b = intervalo
        for nombre, correr in [
            ("Bisección", lambda: raices.biseccion(f, a, b, tol, 500)),
            ("Newton-Raphson", lambda: raices.newton_raphson(f, df, 0.0, tol, 500)),
            ("Secante", lambda: raices.secante(f, 0.0, 1.0, tol, 500)),
        ]:
            r, ms = c.cronometrar(correr)
            resultados[nombre] = r
            ejecuciones.append(c.ejecucion_raiz(nombre, r, t_ref, f, tol=tol, max_iter=500, tiempo_ms=ms, contexto="Tiempo al umbral D(t) = D_u", unidad="meses"))
        tablas.append(c.tabla(
            "Raíces de D(t) − D_u = 0",
            ["Método", "Semillas / intervalo", "t* [meses]", "Iteraciones", "|t − t_ref|", "Error relativo", "|f(t)| residual", "Criterio"],
            [[n, f"[{c.fmt(a)}, {c.fmt(b)}]" if n == "Bisección" else ("t₀ = 0" if n == "Newton-Raphson" else "t₀ = 0, t₁ = 1"),
              c.fmt(r.raiz, 13), r.iteraciones, c.fmt(abs(r.raiz - t_ref)), c.fmt(abs(r.raiz - t_ref) / t_ref if t_ref else None), c.fmt(abs(f(r.raiz))), r.criterio]
             for n, r in resultados.items()],
            f"Referencia: raíz admisible (menor t ≥ 0) de la cuadrática con 50 dígitos: t_ref = {c.fmt(t_ref, 15)} meses. Intervalo por búsqueda incremental de 1 mes.",
        ))
        tablas.append(c.tabla(
            "Iteraciones de Newton-Raphson",
            ["k", "t_k", "f(t_k)", "|t_k − t_(k−1)|", "|t_k − t_ref|"],
            [[h["k"], c.fmt(h["x"], 14), c.fmt(h["fx"]), c.fmt(h["paso"]), c.fmt(abs(h["x"] - t_ref))] for h in resultados["Newton-Raphson"].historial[:60]],
        ))

    # 2) Sensibilidad (problema) y efecto de un mantenimiento retrasado.
    filas_sens = []
    pendiente = df(t_ref) if t_ref is not None else None
    for clave, rel in [("Du", 1e-9), ("Du", 1e-7), ("Du", 0.01), ("alfa", 0.01), ("beta", 0.01)]:
        q = {"D0": D0, "alfa": alfa, "beta": beta, "Du": Du}
        if q[clave] == 0:
            continue
        q[clave] *= 1 + rel
        t_pert = raiz_referencia(q["D0"], q["alfa"], q["beta"], q["Du"])
        if t_ref is None:
            continue
        if t_pert is None:
            filas_sens.append([clave, f"+{rel:g} relativo", c.fmt(t_ref, 10), "sin raíz: el umbral ya no se alcanza", "—", "∞ (cambio cualitativo)"])
        else:
            filas_sens.append([clave, f"+{rel:g} relativo", c.fmt(t_ref, 10), c.fmt(t_pert, 10), c.fmt(t_pert - t_ref), c.fmt(c.kappa_relativo(Du if clave == "Du" else (alfa if clave == "alfa" else beta), q[clave], t_ref, t_pert), 4)])
    kappa_du = abs(Du / (t_ref * pendiente)) if t_ref and pendiente else None
    tablas.append(c.tabla(
        "Sensibilidad del tiempo al umbral",
        ["Parámetro", "Perturbación", "t* [meses]", "t*' [meses]", "Δt", "κ_rel medido"],
        filas_sens,
        f"κ analítico respecto de D_u: |D_u / (t* D′(t*))| = {c.fmt(kappa_du)}; D′(t*) = {c.fmt(pendiente)} %/mes.",
    ))
    retrasos = np.linspace(0, max(p["retraso"], 0.5) * 2, 9)
    exceso = [D(t_ref + d) - Du for d in retrasos] if t_ref is not None else []
    if t_ref is not None:
        tablas.append(c.tabla(
            "Mantenimiento hipotéticamente retrasado",
            ["Retraso [meses]", "Inspección en t [meses]", "D(t) [%]", "Exceso sobre D_u [%]"],
            [[c.fmt(d), c.fmt(t_ref + d, 6), c.fmt(D(t_ref + d), 6), c.fmt(e, 4)] for d, e in zip(retrasos, exceso)],
            "Exceso negativo: la curva vuelve a bajar del umbral (sólo posible con β < 0).",
        ))

    # 3) Sistema lineal de calibración por tramos.
    A, x_exacta, bvec = _sistema(p)
    tol_s = p["tol_sis"]
    max_iter = 5000
    kappa = sistemas.numero_condicion(A)
    rho_j, rho_gs = sistemas.radio_espectral_jacobi(A), sistemas.radio_espectral_gauss_seidel(A)
    historial_res = {}
    filas_sis = []
    for nombre, correr in [
        ("Jacobi", lambda: sistemas.jacobi(A, bvec, tol_s, max_iter)),
        ("Gauss-Seidel", lambda: sistemas.gauss_seidel(A, bvec, tol_s, max_iter)),
        ("LU con pivoteo parcial", lambda: sistemas.resolver_lu(A, bvec)),
        ("Gradiente conjugado", lambda: sistemas.gradiente_conjugado(A, bvec, tol_s, max_iter)),
    ]:
        r, ms = c.cronometrar(correr)
        finito = bool(np.all(np.isfinite(r.x)))
        ea = err.error_absoluto_vector(x_exacta, r.x) if finito else None
        er = err.error_relativo_vector(x_exacta, r.x) if finito else None
        res = err.residual_lineal(A, r.x, bvec) if finito else None
        historial_res[nombre] = r.historial_residual
        filas = []
        for k, (xk, rk) in enumerate(zip(r.historial_x, r.historial_residual), start=1):
            filas.append(c.fila(k, {"x": xk.tolist()}, rk, err.error_absoluto_vector(x_exacta, xk), err.error_relativo_vector(x_exacta, xk)))
        ejecuciones.append(c.ejecucion(
            nombre, "Sistemas lineales", filas,
            referencia=f"x* prefijada, b = A x*; n = {p['n']}", valor={"x": r.x.tolist() if finito else None},
            metricas={"error_abs": ea, "error_rel": er, "residual": res, "iteraciones": r.iteraciones, "kappa": kappa},
            interpretacion=f"{nombre}: {'converge' if r.convergio else 'no converge'} en {r.iteraciones} iteraciones ({r.criterio}); error relativo {c.fmt(er)} con residuo {c.fmt(res)}.",
            convergio=r.convergio, criterio=r.criterio, tiempo_ms=ms, tolerancia=None if nombre.startswith("LU") else tol_s, max_iter=None if nombre.startswith("LU") else max_iter,
            total=r.iteraciones,
        ))
        filas_sis.append([nombre, r.iteraciones, "Sí" if r.convergio else "No", c.fmt(ea), c.fmt(er), c.fmt(res), c.fmt(ms, 3), r.criterio])
    amplif = _amplificacion(A, bvec)
    nombre_matriz = "Hilbert" if p["matriz"] == "hilbert" else "I + " + c.fmt(p["lam"]) + " L"
    tablas.append(c.tabla(
        f"Sistema A x = b ({nombre_matriz}, n = {p['n']})",
        ["Método", "Iter.", "¿Convergió?", "‖x − x*‖₂", "Error relativo", "Residuo relativo", "t [ms]", "Criterio de parada"],
        filas_sis,
        f"κ₂(A) = {c.fmt(kappa)}; ρ(T_J) = {c.fmt(rho_j, 4)}; ρ(T_GS) = {c.fmt(rho_gs, 4)}; SPD = {'sí' if sistemas.es_spd(A) else 'no'}; amplificación medida de δb en x = {c.fmt(amplif)}.",
    ))

    # Indicadores.
    newton = resultados.get("Newton-Raphson")
    indicadores += [
        c.indicador("Tiempo al umbral (Newton)", newton.raiz if newton else None, "meses", f"referencia {c.fmt(t_ref, 10)}" if t_ref is not None else "el umbral no se alcanza"),
        c.indicador("Pendiente D′(t*)", pendiente, "%/mes", "pendiente pequeña ⇒ raíz sensible"),
        c.indicador("κ del tiempo respecto de D_u", kappa_du, "", "analítico"),
        c.indicador(f"Exceso con {c.fmt(p['retraso'])} meses de retraso", D(t_ref + p["retraso"]) - Du if t_ref is not None else None, "%"),
        c.indicador("κ₂(A) del sistema", kappa, "", f"n = {p['n']}"),
        c.indicador("ρ(T_J) / ρ(T_GS)", f"{c.fmt(rho_j, 4)} / {c.fmt(rho_gs, 4)}", "", "< 1 ⇒ el iterativo converge"),
    ]

    # Gráficos.
    t_fin = (t_ref * 1.6 if t_ref else 60.0) + p["retraso"]
    malla = np.linspace(0, t_fin, 200)
    series = [c.serie("D(t)", malla, [D(t) for t in malla], color=1), c.serie("Umbral D_u", [0, t_fin], [Du, Du], color=0, punteada=True)]
    if t_ref is not None:
        series.append(c.serie("t* (raíz)", [t_ref], [Du], estilo="puntos", color=3))
        series.append(c.serie("Inspección retrasada", [t_ref + p["retraso"]], [D(t_ref + p["retraso"])], estilo="puntos", color=2))
    graficos.append(c.grafico("Indicador de deterioro y umbral de inspección", "t [meses]", "D [%]", series))
    if resultados:
        graficos.append(c.grafico(
            "Convergencia de las raíces", "iteración k", "|t_k − t_ref| [meses]",
            [c.serie(n, [h["k"] for h in r.historial], [abs(h["x"] - t_ref) for h in r.historial], estilo="linea-puntos", color=i + 1) for i, (n, r) in enumerate(resultados.items())],
            log_y=True, nota="Bisección: recta (convergencia lineal). Newton: caída cuadrática. Secante: superlineal.",
        ))
    graficos.append(c.grafico(
        "Residuo relativo de los métodos del sistema", "iteración k", "‖b − A x‖/‖b‖",
        [c.serie(n, list(range(1, len(h) + 1)), h, estilo="linea" if len(h) > 30 else "linea-puntos", color=i + 1) for i, (n, h) in enumerate(historial_res.items())],
        log_y=True, nota=f"Tolerancia {tol_s:g}. LU es un método directo (un solo punto).",
    ))
    if t_ref is not None:
        graficos.append(c.grafico("Exceso de deterioro según el retraso del mantenimiento", "retraso [meses]", "D − D_u [%]", [c.serie("Exceso", retrasos, exceso, estilo="linea-puntos", color=2)]))

    # Conclusiones.
    it = {n: r.iteraciones for n, r in resultados.items()}
    s = {f[0]: f for f in filas_sis}
    if tipo == "MAL_CONDICIONADO":
        numerica = (f"Las tres raíces convergen (bisección {it.get('Bisección')}, Newton {it.get('Newton-Raphson')}, secante {it.get('Secante')} iteraciones), "
                    f"pero D′(t*) = {c.fmt(pendiente)} hace que el problema sea mal condicionado: κ = {c.fmt(kappa_du)} y una perturbación relativa de 10⁻⁹ en D_u se amplifica unas κ veces, "
                    f"mientras que +10⁻⁷ elimina la raíz (el umbral deja de alcanzarse). En el sistema de Hilbert (κ₂ = {c.fmt(kappa)}) Jacobi diverge (ρ = {c.fmt(rho_j, 3)} > 1), "
                    "Gauss-Seidel no termina, y LU y gradiente conjugado dejan residuos mínimos con errores mucho mayores: mal condicionamiento y falta de convergencia son fenómenos distintos.")
    else:
        numerica = (f"Newton ({it.get('Newton-Raphson')} iteraciones) y secante ({it.get('Secante')}) alcanzan la raíz de referencia hasta el redondeo; la bisección necesita "
                    f"{it.get('Bisección')} iteraciones porque reduce el intervalo a la mitad en cada paso. En el sistema (κ₂ = {c.fmt(kappa, 4)}), Gauss-Seidel requiere "
                    f"{s['Gauss-Seidel'][1]} iteraciones frente a {s['Jacobi'][1]} de Jacobi (ρ_GS ≈ ρ_J²) y el gradiente conjugado {s['Gradiente conjugado'][1]}; LU es exacto hasta el redondeo.")
    if t_ref is None:
        problema = "Con estos parámetros el indicador no alcanza el umbral: no hay fecha de inspección por umbral."
    else:
        problema = (f"El indicador alcanza el umbral de {c.fmt(Du)} % a los {c.fmt(t_ref, 5)} meses. Un retraso de {c.fmt(p['retraso'])} meses lleva el desgaste a "
                    f"{c.fmt(D(t_ref + p['retraso']), 5)} % ({c.fmt(D(t_ref + p['retraso']) - Du, 3)} % respecto del umbral; negativo indica que la curva ya volvió a bajar). ")
        if tipo == "MAL_CONDICIONADO":
            problema += ("Como la curva casi toca el umbral, la fecha de inspección depende de forma extrema del umbral elegido y de los coeficientes: "
                         "no debe planificarse con un solo valor, sino con un margen o con un umbral menor.")
        else:
            problema += f"La fecha es más sensible al umbral y a la tasa α cuanto menor es la pendiente D′(t*) = {c.fmt(pendiente, 4)} %/mes."
    limites = "D(t) es un indicador hipotético cuadrático; no representa un modelo de desgaste validado ni datos de la empresa. x* se fija para poder medir el error real."

    return {
        "descripcion": ESCENARIOS[tipo]["descripcion"],
        "indicadores": indicadores,
        "ejecuciones": ejecuciones,
        "tablas": tablas,
        "graficos": graficos,
        "conclusiones": {"numerica": numerica, "problema": problema, "limites": limites},
        "panel": [
            ["t al umbral", f"{c.fmt(t_ref, 6)} meses" if t_ref is not None else "no se alcanza"],
            ["κ(t* ; D_u) / κ₂(A)", f"{c.fmt(kappa_du, 3)} / {c.fmt(kappa, 3)}"],
            ["Iter. Jacobi / GS / CG", f"{s['Jacobi'][1]} / {s['Gauss-Seidel'][1]} / {s['Gradiente conjugado'][1]}"],
        ],
    }
