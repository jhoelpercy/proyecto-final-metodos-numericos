"""Módulo 4: reconstrucción de señales de sensores.

Se registra una señal sintética conocida en N instantes uniformes, se ocultan algunos registros
(puntos de prueba) y se reconstruyen con Lagrange, Newton (diferencias divididas) y spline cúbico
natural usando sólo los registros restantes. El error real es |p(t_oculto) − y(t_oculto)|.
El sistema tridiagonal SPD del spline se resuelve con gradiente conjugado (y se contrasta con
Gauss-Seidel y LU). Sobre el spline se buscan cruces de umbral con bisección, Newton y secante.
"""

from __future__ import annotations

import math

import numpy as np

from ..numerico import interpolacion as itp
from ..numerico import raices, sistemas
from . import comun as c

CODIGO = "sensores"
ID = 4
NOMBRE = "Reconstrucción de sensores"

SENALES = {
    "seno": {"nombre": "y(t) = sin t", "intervalo": (0.0, 2 * math.pi), "f": lambda t: np.sin(t), "unidad": "s"},
    "amortiguada": {"nombre": "y(t) = e^(−0.3t) sin 2t", "intervalo": (0.0, 6.0), "f": lambda t: np.exp(-0.3 * np.asarray(t)) * np.sin(2 * np.asarray(t)), "unidad": "s"},
    "runge": {"nombre": "y(t) = 1/(1 + 25t²) (pulso de Runge)", "intervalo": (-1.0, 1.0), "f": lambda t: 1.0 / (1.0 + 25.0 * np.asarray(t) ** 2), "unidad": "s"},
}

PARAMETROS = [
    {"clave": "senal", "etiqueta": "Señal sintética", "unidad": "", "opciones": list(SENALES)},
    {"clave": "N", "etiqueta": "Registros N (uniformes)", "unidad": "", "min": 5, "max": 61, "entero": True, "paso": 1},
    {"clave": "sigma", "etiqueta": "Ruido gaussiano σ", "unidad": "u", "min": 0, "max": 0.5, "paso": "any"},
    {"clave": "semilla", "etiqueta": "Semilla del ruido", "unidad": "", "min": 0, "max": 1000000, "entero": True, "paso": 1},
    {"clave": "cada", "etiqueta": "Ocultar uno de cada k", "unidad": "", "min": 2, "max": 10, "entero": True, "paso": 1},
    {"clave": "desfase", "etiqueta": "Desfase de los ocultos", "unidad": "", "min": 1, "max": 9, "entero": True, "paso": 1},
    {"clave": "u", "etiqueta": "Umbral de cruce", "unidad": "u", "min": -1, "max": 1, "paso": "any"},
    {"clave": "tol", "etiqueta": "Tolerancia de raíces", "unidad": "", "min": 1e-15, "max": 1e-2, "paso": "any"},
]

ESCENARIOS = {
    "NORMAL": {
        "descripcion": "Seno con 17 registros sin ruido; se ocultan 4 registros interiores.",
        "parametros": {"senal": "seno", "N": 17, "sigma": 0.0, "semilla": 1, "cada": 4, "desfase": 2, "u": 0.5, "tol": 1e-12},
        "fuente": "Sintético: y(t) = sin t con valores de referencia exactos.",
    },
    "ESTRES": {
        "descripcion": "Senoidal amortiguada con 31 registros, ruido σ = 0.002 (semilla 7) y 6 registros ocultos: polinomio de grado 24.",
        "parametros": {"senal": "amortiguada", "N": 31, "sigma": 0.002, "semilla": 7, "cada": 5, "desfase": 2, "u": 0.0, "tol": 1e-12},
        "fuente": "Sintético: ruido gaussiano reproducible con semilla fija.",
    },
    "MAL_CONDICIONADO": {
        "descripcion": "Subexperimento numérico: pulso de Runge con 41 registros, se pierden los impares y quedan 21 nodos uniformes (distribución desfavorable).",
        "parametros": {"senal": "runge", "N": 41, "sigma": 0.0, "semilla": 1, "cada": 2, "desfase": 1, "u": 0.5, "tol": 1e-12},
        "fuente": "Sintético: función de Runge, ejemplo clásico de nodos uniformes mal condicionados.",
    },
}


def _datos(p: dict):
    s = SENALES[p["senal"]]
    a, b = s["intervalo"]
    t = np.linspace(a, b, p["N"])
    y_real = s["f"](t)
    rng = np.random.default_rng(p["semilla"])
    y_obs = y_real + p["sigma"] * rng.standard_normal(len(t))
    ocultos = np.array([i for i in range(1, p["N"] - 1) if i % p["cada"] == p["desfase"] % p["cada"]], dtype=int)
    if len(ocultos) == 0:
        raise c.ErrorParametros(["Con ese k y desfase no se oculta ningún registro interior."])
    nodos = np.setdiff1d(np.arange(p["N"]), ocultos)
    if len(nodos) < 4:
        raise c.ErrorParametros(["Quedan menos de 4 nodos para interpolar."])
    return s, t, y_real, y_obs, ocultos, nodos


def _reconstrucciones(tn, yn):
    coef, tabla = itp.diferencias_divididas(tn, yn)
    spline = itp.spline_natural(tn, yn, "Gradiente conjugado")
    return {
        "Lagrange": lambda x: itp.lagrange(tn, yn, x),
        "Newton (diferencias divididas)": lambda x: itp.newton_evaluar(tn, coef, x),
        "Spline cúbico natural": lambda x: spline(x),
    }, spline, coef, tabla


def _primer_cruce(f, a, b, n=4000):
    malla = np.linspace(a, b, n + 1)
    valores = f(malla)
    for i in range(n):
        if valores[i] * valores[i + 1] < 0:
            return float(malla[i]), float(malla[i + 1])
    return None


def ejecutar(tipo: str, p: dict) -> dict:
    s, t, y_real, y_obs, ocultos, nodos = _datos(p)
    a, b = s["intervalo"]
    tn, yn = t[nodos], y_obs[nodos]
    th, yh = t[ocultos], y_real[ocultos]
    densa = np.linspace(a, b, 2001)
    y_densa = s["f"](densa)
    metodos, spline, coef, tabla_dd = _reconstrucciones(tn, yn)
    metodos_limpios, _, _, _ = _reconstrucciones(tn, y_real[nodos])
    ejecuciones, tablas, graficos = [], [], []

    # 1) Interpolación: error en los registros ocultos y en una malla densa.
    resumen = {}
    for nombre, p_ in metodos.items():
        (pred, ms) = c.cronometrar(lambda: p_(th))
        e = np.abs(pred - yh)
        e_densa = np.abs(p_(densa) - y_densa)
        ruido = float(np.max(np.abs(p_(densa) - metodos_limpios[nombre](densa)))) if p["sigma"] > 0 else 0.0
        resumen[nombre] = {"pred": pred, "max_oculto": float(e.max()), "rms_oculto": float(np.sqrt(np.mean(e**2))), "max_densa": float(e_densa.max()), "ruido": ruido, "e_densa": e_densa}
        filas = [c.fila(int(i), {"t": float(ti), "y_ref": float(yr), "y_aprox": float(pv)}, None, float(ei), float(ei / abs(yr)) if abs(yr) > 1e-12 else None)
                 for i, ti, yr, pv, ei in zip(ocultos, th, yh, pred, e)]
        extra = {}
        if nombre.startswith("Spline"):
            extra = {"iteraciones_sistema": spline.iteraciones_sistema, "residuo_sistema": spline.residuo_sistema, "kappa_sistema": spline.kappa_sistema}
        ejecuciones.append(c.ejecucion(
            nombre, "Interpolación", filas,
            referencia=f"{s['nombre']} evaluada en los {len(ocultos)} registros ocultos",
            valor={"predicciones": pred.tolist(), "nodos": len(nodos), "grado": len(nodos) - 1 if not nombre.startswith("Spline") else 3},
            metricas={"error_max_oculto": float(e.max()), "rms_oculto": resumen[nombre]["rms_oculto"], "error_max_malla": float(e_densa.max()), "efecto_ruido": ruido, **extra},
            interpretacion=f"{nombre}: error máximo {c.fmt(e.max())} en los registros ocultos y {c.fmt(e_densa.max())} en todo el intervalo.",
            tiempo_ms=ms,
        ))
    tablas.append(c.tabla(
        "Reconstrucción de los registros ocultos",
        ["i", "t", "y real", "Lagrange", "|e|", "Newton", "|e|", "Spline", "|e|"],
        [[int(i), c.fmt(ti, 6), c.fmt(yr, 8)] + sum(([c.fmt(resumen[n]["pred"][k], 8), c.fmt(abs(resumen[n]["pred"][k] - yr))] for n in metodos), [])
         for k, (i, ti, yr) in enumerate(zip(ocultos, th, yh))],
        f"Se interpola con {len(nodos)} registros (polinomio de grado {len(nodos) - 1}); la referencia es la señal sin ruido.",
    ))
    diferencia_LN = float(np.max(np.abs(metodos["Lagrange"](densa) - metodos["Newton (diferencias divididas)"](densa))))
    tablas.append(c.tabla(
        "Resumen de errores por método",
        ["Método", "max |e| ocultos", "RMS ocultos", "max |e| en [a, b]", "Efecto del ruido max|p_σ − p_0|"],
        [[n, c.fmt(r["max_oculto"]), c.fmt(r["rms_oculto"]), c.fmt(r["max_densa"]), c.fmt(r["ruido"]) if p["sigma"] > 0 else "sin ruido"] for n, r in resumen.items()],
        f"Lagrange y Newton son el mismo polinomio: su diferencia máxima en la malla es {c.fmt(diferencia_LN)} (sólo redondeo).",
    ))
    k_dd = min(8, len(coef))
    tablas.append(c.tabla(
        "Diferencias divididas (primeras columnas)",
        ["i", "t_i"] + [f"f[·]^{j}" for j in range(k_dd)],
        [[i, c.fmt(tn[i], 6)] + [c.fmt(tabla_dd[i, j], 6) if not np.isnan(tabla_dd[i, j]) else "" for j in range(k_dd)] for i in range(min(len(tn), 12))],
        "Los coeficientes del polinomio de Newton son la primera fila.",
    ))

    # 2) Condicionamiento: constante de Lebesgue, Vandermonde y nodos de Chebyshev.
    malla_l = np.linspace(a, b, 4001)
    lam_u = itp.constante_lebesgue(tn, malla_l)
    n_nodos = len(nodos)
    cheb = (a + b) / 2 + (b - a) / 2 * np.cos((2 * np.arange(n_nodos) + 1) * np.pi / (2 * n_nodos))[::-1]
    lam_c = itp.constante_lebesgue(cheb, malla_l)
    esc = lambda x: (2 * x - (a + b)) / (b - a)
    kappa_vu = float(np.linalg.cond(np.vander(esc(tn), increasing=True)))
    kappa_vc = float(np.linalg.cond(np.vander(esc(cheb), increasing=True)))
    coef_c, _ = itp.diferencias_divididas(cheb, s["f"](cheb))
    e_cheb = np.abs(itp.newton_evaluar(cheb, coef_c, densa) - y_densa)
    delta = 1e-6
    yp = yn + delta * np.sign(np.sin(np.arange(len(yn)) * 1.7 + 0.3))
    amplif = float(np.max(np.abs(itp.lagrange(tn, yp, densa) - itp.lagrange(tn, yn, densa)))) / delta
    tablas.append(c.tabla(
        "Condicionamiento de la interpolación polinómica",
        ["Nodos", "n", "Constante de Lebesgue Λ", "κ₂(Vandermonde)", "max |p − y|"],
        [["Registros disponibles", n_nodos, c.fmt(lam_u), c.fmt(kappa_vu), c.fmt(resumen["Lagrange"]["max_densa"])],
         ["Chebyshev (mismo n, remuestreo ideal)", n_nodos, c.fmt(lam_c), c.fmt(kappa_vc), c.fmt(float(e_cheb.max()))]],
        f"Una perturbación de ±{delta:g} en los datos cambió el polinomio hasta {c.fmt(amplif * delta)} (amplificación {c.fmt(amplif)} ≤ Λ). "
        "Λ grande indica mal condicionamiento por la ubicación de los nodos, no por el algoritmo.",
    ))

    # 3) Sistema auxiliar del spline.
    A, bb = itp.sistema_spline_natural(tn, yn)
    filas_sis = []
    if np.any(bb):
        ref = np.linalg.solve(A, bb)
        for nombre, correr in [
            ("Gradiente conjugado", lambda: sistemas.gradiente_conjugado(A, bb, 1e-13)),
            ("Gauss-Seidel", lambda: sistemas.gauss_seidel(A, bb, 1e-13, 5000)),
            ("Jacobi", lambda: sistemas.jacobi(A, bb, 1e-13, 5000)),
            ("LU con pivoteo parcial", lambda: sistemas.resolver_lu(A, bb)),
        ]:
            r, ms = c.cronometrar(correr)
            filas_sis.append([nombre, r.iteraciones, "Sí" if r.convergio else "No", c.fmt(float(np.linalg.norm(r.x - ref) / np.linalg.norm(ref))), c.fmt(r.historial_residual[-1] if r.historial_residual else None), c.fmt(ms, 3)])
    tablas.append(c.tabla(
        f"Sistema tridiagonal del spline ({A.shape[0]} momentos interiores)",
        ["Método", "Iteraciones", "¿Convergió?", "Error relativo vs numpy.linalg.solve", "Residuo relativo", "t [ms]"],
        filas_sis,
        f"Matriz simétrica, estrictamente diagonal dominante con diagonal positiva ⇒ SPD; κ₂ = {c.fmt(spline.kappa_sistema, 4)}. Jacobi y Gauss-Seidel convergen por dominancia diagonal.",
    ))

    # 4) Cruce de umbral sobre el spline.
    u, tol = p["u"], p["tol"]
    g_real = lambda x: float(s["f"](x)) - u
    cruce_real = _primer_cruce(lambda x: s["f"](x) - u, a, b)
    g = lambda x: float(spline(x)[0]) - u
    dg = lambda x: float(spline.derivada(x)[0])
    cruce_s = _primer_cruce(lambda x: spline(x) - u, a, b)
    resultados_raiz = {}
    t_ref = None
    if cruce_real and cruce_s:
        t_ref = raices.biseccion(g_real, *cruce_real, tol=1e-15, max_iter=200).raiz
        ia, ib = cruce_s
        for nombre, correr in [
            ("Bisección", lambda: raices.biseccion(g, ia, ib, tol, 300)),
            ("Newton-Raphson", lambda: raices.newton_raphson(g, dg, ia, tol, 100)),
            ("Secante", lambda: raices.secante(g, ia, ib, tol, 100)),
        ]:
            r, ms = c.cronometrar(correr)
            resultados_raiz[nombre] = r
            ejecuciones.append(c.ejecucion_raiz(nombre, r, t_ref, g, tol=tol, max_iter=300, tiempo_ms=ms, contexto=f"Primer cruce del spline con y = {c.fmt(u)}", unidad="s"))
        tablas.append(c.tabla(
            f"Primer cruce del umbral y = {c.fmt(u)} sobre el spline reconstruido",
            ["Método", "Intervalo / semillas", "t cruce", "Iteraciones", "|t − t_real|", "|s(t) − u|", "Criterio"],
            [[n, f"[{c.fmt(ia, 6)}, {c.fmt(ib, 6)}]" if n != "Newton-Raphson" else f"t₀ = {c.fmt(ia, 6)}", c.fmt(r.raiz, 12), r.iteraciones,
              c.fmt(abs(r.raiz - t_ref)), c.fmt(abs(g(r.raiz))), r.criterio] for n, r in resultados_raiz.items()],
            f"Cruce de la señal real (bisección a 1e−15): t = {c.fmt(t_ref, 14)}. La diferencia con el spline es error de reconstrucción, no de la raíz.",
        ))

    # Indicadores.
    mejor = min(resumen, key=lambda n: resumen[n]["max_oculto"])
    indicadores = [
        c.indicador("Registros ocultos / nodos usados", f"{len(ocultos)} / {len(nodos)}"),
        c.indicador("Mejor método en ocultos", mejor, "", f"max |e| = {c.fmt(resumen[mejor]['max_oculto'])}"),
        c.indicador("max |e| Lagrange / Spline", f"{c.fmt(resumen['Lagrange']['max_densa'], 3)} / {c.fmt(resumen['Spline cúbico natural']['max_densa'], 3)}", "", "en todo el intervalo"),
        c.indicador("Constante de Lebesgue Λ", lam_u, "", f"Chebyshev: {c.fmt(lam_c, 4)}"),
        c.indicador("Iteraciones CG del spline", spline.iteraciones_sistema, "", f"κ₂ = {c.fmt(spline.kappa_sistema, 4)}"),
    ]
    if t_ref is not None and resultados_raiz:
        indicadores.append(c.indicador("Cruce del umbral (spline)", resultados_raiz["Newton-Raphson"].raiz, "s", f"real {c.fmt(t_ref, 8)} s"))

    # Gráficos.
    def recorte(v):
        return [float(x) if abs(x) < 5 else None for x in v]

    series = [c.serie("Señal real", densa, y_densa, color=0)]
    for i, (n, p_) in enumerate(metodos.items()):
        series.append(c.serie(n.split(" (")[0], densa, recorte(p_(densa)), color=i + 1, punteada=n.startswith("Newton")))
    series.append(c.serie("Registros usados", tn, yn, estilo="puntos", color=0))
    series.append(c.serie("Registros ocultos (real)", th, yh, estilo="puntos", color=4))
    graficos.append(c.grafico("Señal y reconstrucciones", "t [s]", "y [u]", series, nota="Valores con |p| > 5 se recortan del gráfico para no aplastar la escala."))
    graficos.append(c.grafico(
        "Error de reconstrucción en todo el intervalo", "t [s]", "|p(t) − y(t)|",
        [c.serie(n.split(" (")[0], densa, r["e_densa"], color=i + 1, punteada=n.startswith("Newton")) for i, (n, r) in enumerate(resumen.items())]
        + [c.serie("Polinomio en nodos de Chebyshev", densa, e_cheb, color=4, punteada=True)],
        log_y=True, nota="Los mínimos (error ≈ 0) coinciden con los nodos.",
    ))
    graficos.append(c.grafico(
        "Error en los registros ocultos", "t [s]", "|e|",
        [c.serie(n.split(" (")[0], th, np.abs(r["pred"] - yh), estilo="linea-puntos", color=i + 1) for i, (n, r) in enumerate(resumen.items())],
        log_y=True,
    ))
    if resultados_raiz:
        graficos.append(c.grafico(
            "Convergencia del cruce de umbral", "iteración k", "|t_k − t*| (t* del spline)",
            [c.serie(n, [h["k"] for h in r.historial], [abs(h["x"] - resultados_raiz["Bisección"].raiz) for h in r.historial], estilo="linea-puntos", color=i + 1) for i, (n, r) in enumerate(resultados_raiz.items())],
            log_y=True,
        ))

    # Conclusiones.
    rl, rs_ = resumen["Lagrange"], resumen["Spline cúbico natural"]
    if tipo == "MAL_CONDICIONADO":
        numerica = (f"Con {n_nodos} nodos uniformes, Λ = {c.fmt(lam_u, 4)} y κ₂(Vandermonde) = {c.fmt(kappa_vu, 3)}: el polinomio interpolante oscila cerca de los extremos "
                    f"(error {c.fmt(rl['max_densa'], 3)}) aunque Lagrange y Newton coinciden hasta {c.fmt(diferencia_LN, 2)}. No es un fallo de los algoritmos: con nodos de Chebyshev "
                    f"(Λ = {c.fmt(lam_c, 3)}) el mismo grado da {c.fmt(float(e_cheb.max()), 3)}, y el spline, de grado bajo por tramos, da {c.fmt(rs_['max_densa'], 3)}.")
    elif tipo == "ESTRES":
        numerica = (f"Con ruido σ = {p['sigma']} y grado {n_nodos - 1}, el polinomio amplifica el ruido hasta {c.fmt(rl['ruido'], 3)} (Λ = {c.fmt(lam_u, 3)}), "
                    f"mientras el spline lo deja en {c.fmt(rs_['ruido'], 3)}. En los registros ocultos el spline logra {c.fmt(rs_['max_oculto'], 3)} frente a {c.fmt(rl['max_oculto'], 3)} del polinomio. "
                    "El sistema del spline converge con CG, Gauss-Seidel y Jacobi por ser diagonal dominante.")
    else:
        numerica = (f"Sin ruido y con pocos nodos, el polinomio de grado {n_nodos - 1} es muy preciso (error {c.fmt(rl['max_oculto'], 3)} en los ocultos) y supera al spline "
                    f"({c.fmt(rs_['max_oculto'], 3)}), porque sin t es analítica y sus derivadas están acotadas. Lagrange y Newton difieren sólo en {c.fmt(diferencia_LN, 2)}; "
                    "Newton permite agregar nodos sin recalcular todo y evaluar con Horner.")
    problema = (f"Para el sensor, la reconstrucción de los {len(ocultos)} registros perdidos es confiable con {mejor.split(' (')[0]} (error máximo {c.fmt(resumen[mejor]['max_oculto'], 3)} u). ")
    if p["sigma"] > 0:
        problema += "Con ruido, un polinomio de grado alto no debe usarse para rellenar huecos: convierte pequeñas perturbaciones en grandes errores; el spline es la opción robusta. "
    if tipo == "MAL_CONDICIONADO":
        problema += "Si el muestreo es uniforme y la señal tiene un pulso, sólo el spline (o remuestrear en nodos de Chebyshev) da una reconstrucción útil cerca de los bordes. "
    if t_ref is not None and resultados_raiz:
        problema += f"El cruce del umbral se estima en t = {c.fmt(resultados_raiz['Newton-Raphson'].raiz, 6)} s frente a {c.fmt(t_ref, 6)} s reales."
    limites = "Señales sintéticas con forma conocida y ruido gaussiano independiente; no se modelan derivas, saturación ni pérdidas en ráfaga de sensores reales."

    return {
        "descripcion": ESCENARIOS[tipo]["descripcion"],
        "indicadores": indicadores,
        "ejecuciones": ejecuciones,
        "tablas": tablas,
        "graficos": graficos,
        "conclusiones": {"numerica": numerica, "problema": problema, "limites": limites},
        "panel": [
            ["max |e| ocultos L / S", f"{c.fmt(rl['max_oculto'], 2)} / {c.fmt(rs_['max_oculto'], 2)}"],
            ["Λ uniformes / Chebyshev", f"{c.fmt(lam_u, 3)} / {c.fmt(lam_c, 3)}"],
            ["Cruce de umbral", f"{c.fmt(resultados_raiz['Newton-Raphson'].raiz, 6)} s" if resultados_raiz else "sin cruce"],
        ],
    }
