"""Módulo 3: oscilaciones por viento.

Oscilador lineal de un grado de libertad:
    m x″ + c x′ + k x = F(t),   F(t) = F0 sin(Ω t),   c = 2 ζ √(k m)
    x′ = v,   v′ = (F(t) − c v − k x)/m.
Referencia cerrada (0 < ζ < 1): homogénea subamortiguada + particular armónica
    x(t) = e^(−ζω₀t)[C₁ cos(ω_d t) + C₂ sin(ω_d t)] + X sin(Ω t − φ).
Magnitud integrada: I = ∫₀^T x(t)² dt [m²·s], referencia por Gauss-Legendre compuesto sobre la solución cerrada.
"""

from __future__ import annotations

import math

import numpy as np

from ..numerico import edo, integracion
from . import comun as c

CODIGO = "oscilaciones"
ID = 3
NOMBRE = "Oscilaciones por viento"

PARAMETROS = [
    {"clave": "m", "etiqueta": "Masa de la cabina m", "unidad": "kg", "min": 100, "max": 20000, "paso": "any"},
    {"clave": "k", "etiqueta": "Rigidez equivalente k", "unidad": "N/m", "min": 10, "max": 1e6, "paso": "any"},
    {"clave": "zeta", "etiqueta": "Amortiguamiento relativo ζ", "unidad": "", "min": 0.0005, "max": 0.99, "paso": "any"},
    {"clave": "F0", "etiqueta": "Amplitud de la ráfaga F₀", "unidad": "N", "min": 0, "max": 1e5, "paso": "any"},
    {"clave": "r", "etiqueta": "Frecuencia relativa Ω/ω₀", "unidad": "", "min": 0.05, "max": 5, "paso": "any"},
    {"clave": "x0", "etiqueta": "Desplazamiento inicial x₀", "unidad": "m", "min": -5, "max": 5, "paso": "any"},
    {"clave": "v0", "etiqueta": "Velocidad inicial v₀", "unidad": "m/s", "min": -5, "max": 5, "paso": "any"},
    {"clave": "T", "etiqueta": "Tiempo simulado T", "unidad": "s", "min": 1, "max": 600, "paso": "any"},
    {"clave": "h", "etiqueta": "Paso de la EDO h", "unidad": "s", "min": 0.005, "max": 2, "paso": "any"},
]

ESCENARIOS = {
    "NORMAL": {
        "descripcion": "Vibración libre subamortiguada tras un desplazamiento de 0.1 m (F = 0): caso analítico de referencia.",
        "parametros": {"m": 1200.0, "k": 4800.0, "zeta": 0.05, "F0": 0.0, "r": 0.95, "x0": 0.1, "v0": 0.0, "T": 20.0, "h": 0.05},
        "fuente": "Sintético: m y k supuestos (ω₀ = 2 rad/s), ζ típico de estructuras ligeras.",
    },
    "ESTRES": {
        "descripcion": "Ráfaga armónica intensa F₀ = 1500 N cerca de la resonancia (Ω = 0.95 ω₀), 60 s simulados con paso grande h = 0.5 s.",
        "parametros": {"m": 1200.0, "k": 4800.0, "zeta": 0.05, "F0": 1500.0, "r": 0.95, "x0": 0.0, "v0": 0.0, "T": 60.0, "h": 0.5},
        "fuente": "Sintético: forzamiento armónico idealizado del viento.",
    },
    "MAL_CONDICIONADO": {
        "descripcion": "Subexperimento numérico: ζ = 0.005 y Ω en el flanco de la resonancia (Ω = 0.995 ω₀), donde la amplitud es muy sensible a Ω.",
        "parametros": {"m": 1200.0, "k": 4800.0, "zeta": 0.005, "F0": 50.0, "r": 0.995, "x0": 0.0, "v0": 0.0, "T": 300.0, "h": 0.05},
        "fuente": "Sintético: amortiguamiento muy bajo elegido para exhibir la sensibilidad de la respuesta.",
    },
}

GL_NODOS, GL_PESOS = np.polynomial.legendre.leggauss(10)


class Analitica:
    def __init__(self, p: dict):
        self.p = p
        self.m, self.k, self.zeta = p["m"], p["k"], p["zeta"]
        self.c = 2 * self.zeta * math.sqrt(self.k * self.m)
        self.w0 = math.sqrt(self.k / self.m)
        self.wd = self.w0 * math.sqrt(1 - self.zeta**2)
        self.sigma = self.zeta * self.w0
        self.W = p["r"] * self.w0
        self.F0 = p["F0"]
        self.X, self.phi = amplitud_permanente(p)
        self.C1 = p["x0"] + self.X * math.sin(self.phi)
        self.C2 = (p["v0"] - self.X * self.W * math.cos(self.phi) + self.sigma * self.C1) / self.wd

    def x(self, t):
        t = np.asarray(t, float)
        return np.exp(-self.sigma * t) * (self.C1 * np.cos(self.wd * t) + self.C2 * np.sin(self.wd * t)) + self.X * np.sin(self.W * t - self.phi)

    def v(self, t):
        t = np.asarray(t, float)
        a = -self.sigma * self.C1 + self.wd * self.C2
        b = -self.sigma * self.C2 - self.wd * self.C1
        return np.exp(-self.sigma * t) * (a * np.cos(self.wd * t) + b * np.sin(self.wd * t)) + self.X * self.W * np.cos(self.W * t - self.phi)

    def campo(self, t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], (self.F0 * math.sin(self.W * t) - self.c * y[1] - self.k * y[0]) / self.m])

    def integral_x2(self, T: float) -> float:
        """Gauss-Legendre de 10 puntos compuesto: referencia independiente de las reglas evaluadas."""
        paneles = int(max(2000, 40 * T))
        bordes = np.linspace(0, T, paneles + 1)
        medio, semi = (bordes[1:] + bordes[:-1]) / 2, (bordes[1:] - bordes[:-1]) / 2
        t = medio[:, None] + semi[:, None] * GL_NODOS[None, :]
        return float(np.sum(semi[:, None] * GL_PESOS[None, :] * self.x(t) ** 2))


def amplitud_permanente(p: dict) -> tuple[float, float]:
    w0 = math.sqrt(p["k"] / p["m"])
    W = p["r"] * w0
    g = (w0**2 - W**2) ** 2 + (2 * p["zeta"] * w0 * W) ** 2
    return (p["F0"] / p["m"]) / math.sqrt(g), math.atan2(2 * p["zeta"] * w0 * W, w0**2 - W**2)


def kappa_amplitud_frecuencia(p: dict) -> float:
    """κ = |d ln X / d ln Ω| analítico."""
    w0 = math.sqrt(p["k"] / p["m"])
    W = p["r"] * w0
    g = (w0**2 - W**2) ** 2 + (2 * p["zeta"] * w0 * W) ** 2
    return abs(2 * W**2 * (w0**2 - W**2) - 4 * p["zeta"] ** 2 * w0**2 * W**2) / g


def factor_amplificacion(metodo: str, z: complex) -> float:
    if metodo == "RK4":
        return abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    return abs(1 + z + z**2 / 2)


def ejecutar(tipo: str, p: dict) -> dict:
    if p["T"] / p["h"] > 40000:
        raise c.ErrorParametros(["T/h no debe superar 40 000 pasos (tiempo de cómputo)."])
    an = Analitica(p)
    T, h = p["T"], p["h"]
    ejecuciones, tablas, graficos = [], [], []
    I_ref = an.integral_x2(T)
    malla_densa = np.linspace(0, T, 200001)
    max_ref = float(np.max(np.abs(an.x(malla_densa))))
    lam = complex(-an.sigma, an.wd)  # autovalor del sistema homogéneo

    # 1) EDO.
    sol, resumen_edo = {}, {}
    for metodo in ("Heun (RK2)", "RK4"):
        r, ms = c.cronometrar(lambda: edo.integrar(an.campo, [p["x0"], p["v0"]], 0.0, T, h, metodo, limite=1e6))
        sol[metodo] = r
        xr = an.x(r.t)
        e = np.abs(r.y[:, 0] - xr)
        filas = [c.fila(i, {"t": ti, "x": xi, "v": vi, "x_ref": xri}, None, float(ei), (float(ei) / abs(xri)) if abs(xri) > 1e-9 else None)
                 for i, (ti, (xi, vi), xri, ei) in enumerate(zip(r.t, r.y, xr, e))]
        amp = float(np.max(np.abs(r.y[:, 0])))
        R = factor_amplificacion(metodo, h * lam)
        resumen_edo[metodo] = {"error_max": float(e.max()), "amp": amp, "R": R}
        ejecuciones.append(c.ejecucion(
            metodo, "EDO", filas,
            referencia="solución cerrada subamortiguada + particular armónica",
            valor={"x_final": float(r.y[-1, 0]), "amplitud_max": amp, "pasos": r.pasos},
            metricas={"error_max_x": float(e.max()), "error_final_x": float(e[-1]), "error_amplitud": abs(amp - max_ref), "factor_amplificacion": R},
            interpretacion=f"{metodo} con h = {h} s: error máximo en x de {c.fmt(e.max())} m; |R(hλ)| = {c.fmt(R, 6)} ({'estable' if R <= 1 else 'amplifica la oscilación'}).",
            convergio=r.estable, criterio=r.criterio, tiempo_ms=ms, paso=h, total=r.pasos,
        ))

    # 2) Convergencia al refinar h.
    pasos = [h, h / 2, h / 4]
    conv = {}
    for metodo in sol:
        datos = []
        for hh in pasos:
            r = sol[metodo] if hh == h else edo.integrar(an.campo, [p["x0"], p["v0"]], 0.0, T, hh, metodo, limite=1e6)
            datos.append((hh, float(np.max(np.abs(r.y[:, 0] - an.x(r.t))))))
        conv[metodo] = datos
    tablas.append(c.tabla(
        "Refinamiento del paso: error máximo en x(t)",
        ["Método", "h [s]", "ω₀h", "|R(hλ)|", "max |x − x_ref| [m]", "orden p"],
        [[met, c.fmt(hh), c.fmt(an.w0 * hh, 4), c.fmt(factor_amplificacion(met, hh * lam), 8), c.fmt(e), c.fmt(c.orden(d[i - 1][1], e), 3) if i else "—"]
         for met, d in conv.items() for i, (hh, e) in enumerate(d)],
        "|R(hλ)| > 1 indica que el método amplifica artificialmente la oscilación libre (λ = −ζω₀ + iω_d).",
    ))

    # 3) Magnitud integrada I = ∫ x² dt.
    x2 = lambda t: float(an.x(t)) ** 2
    rom, ms_rom = c.cronometrar(lambda: integracion.romberg(x2, 0, T, tol=1e-12 * max(1.0, I_ref), max_niveles=16))
    filas_ref, previo = [], {}
    series_int = {"Trapecio": [], "Simpson 1/3": []}
    n_lista = [2**j for j in range(6, 15)]
    for n in n_lista:
        vt, vs = integracion.trapecio(x2, 0, T, n), integracion.simpson(x2, 0, T, n)
        et, es = abs(vt - I_ref), abs(vs - I_ref)
        filas_ref.append([n, c.fmt(T / n), c.fmt(vt, 12), c.fmt(et), c.fmt(c.orden(previo.get("t"), et), 3), c.fmt(vs, 12), c.fmt(es), c.fmt(c.orden(previo.get("s"), es), 3)])
        previo = {"t": et, "s": es}
        series_int["Trapecio"].append((n, et, vt))
        series_int["Simpson 1/3"].append((n, es, vs))
    tablas.append(c.tabla(
        "Refinamiento de I = ∫₀^T x(t)² dt sobre la solución cerrada",
        ["n", "h [s]", "Trapecio", "Error", "orden", "Simpson 1/3", "Error", "orden"],
        filas_ref,
        f"Referencia I = {c.fmt(I_ref, 14)} m²·s (Gauss-Legendre de 10 puntos compuesto). Con pocos subintervalos por periodo aún no hay régimen asintótico. "
        f"Romberg: {c.fmt(rom.valor, 14)} con {rom.evaluaciones} evaluaciones, error {c.fmt(abs(rom.valor - I_ref))} ({rom.criterio}).",
    ))
    y_rk4 = sol["RK4"].y[:, 0] ** 2
    h_malla = T / (len(y_rk4) - 1)
    uniforme = abs(sol["RK4"].t[1] - sol["RK4"].t[0] - h_malla) < 1e-12
    sobre_rk4 = {
        "Trapecio compuesto": integracion.trapecio_datos(y_rk4, h_malla),
        "Simpson 1/3 compuesto": integracion.simpson_datos(y_rk4, h_malla),
        "Romberg": integracion.romberg_datos(y_rk4, h_malla).valor,
    }
    for nombre, clave in [("Trapecio compuesto", "Trapecio"), ("Simpson 1/3 compuesto", "Simpson 1/3")]:
        filas = [c.fila(i + 1, {"n": n, "valor": v}, None, e, e / I_ref if I_ref else None) for i, (n, e, v) in enumerate(series_int[clave])]
        ea, er = c.errores(I_ref, sobre_rk4[nombre])
        ejecuciones.append(c.ejecucion(
            nombre, "Integración", filas, referencia=f"I_ref = {c.fmt(I_ref, 12)} m²·s",
            valor={"I_muestras_rk4": sobre_rk4[nombre], "I_analitica_n_max": series_int[clave][-1][2]},
            metricas={"error_abs_muestras_rk4": ea, "error_rel_muestras_rk4": er, "error_abs_refinado": series_int[clave][-1][1]},
            interpretacion=f"{nombre} sobre las muestras de RK4: I = {c.fmt(sobre_rk4[nombre], 10)} (error {c.fmt(ea)}), que combina error de la EDO y de la cuadratura.",
            paso=h_malla,
        ))
    ea, er = c.errores(I_ref, rom.valor)
    ejecuciones.append(c.ejecucion(
        "Romberg", "Integración",
        [c.fila(i, {"n": 2**i, "diagonal": f_[-1]}, None, *c.errores(I_ref, f_[-1])) for i, f_ in enumerate(rom.tabla)],
        referencia=f"I_ref = {c.fmt(I_ref, 12)} m²·s", valor={"I_analitica": rom.valor, "I_muestras_rk4": sobre_rk4["Romberg"], "niveles": rom.niveles},
        metricas={"error_abs": ea, "error_rel": er, "evaluaciones": rom.evaluaciones, "error_abs_muestras_rk4": abs(sobre_rk4["Romberg"] - I_ref)},
        interpretacion=f"Romberg sobre x(t)² cerrada: {c.fmt(rom.valor, 12)} con {rom.evaluaciones} evaluaciones ({rom.criterio}).",
        convergio=rom.convergio, criterio=rom.criterio, tiempo_ms=ms_rom, tolerancia=1e-12 * max(1.0, I_ref), max_iter=16,
    ))
    tablas.append(c.tabla(
        "Magnitud integrada sobre las muestras de RK4 (aplicación)",
        ["Regla", "I [m²·s]", "|I − I_ref|", "Error relativo"],
        [[n, c.fmt(v, 12), c.fmt(abs(v - I_ref)), c.fmt(abs(v - I_ref) / I_ref if I_ref else None)] for n, v in sobre_rk4.items()],
        f"Malla de RK4 con h = {c.fmt(h_malla)} s{'' if uniforme else ' (último paso recortado)'}; Romberg usa los trapecios con pasos h·2^j que caben en la malla.",
    ))

    # 4) Sensibilidad física (amplitud) y mal condicionamiento.
    kappa_W = kappa_amplitud_frecuencia(p) if p["F0"] > 0 else None
    filas_sens = []
    for clave, rel in [("r", 1e-3), ("zeta", 0.01), ("F0", 0.01), ("k", 0.01)]:
        if p[clave] == 0:
            continue
        q = dict(p, **{clave: p[clave] * (1 + rel)})
        X0, _ = amplitud_permanente(p)
        X1, _ = amplitud_permanente(q)
        I1 = Analitica(q).integral_x2(T)
        filas_sens.append([clave, f"+{rel:g}", c.fmt(X0, 8), c.fmt(X1, 8), c.fmt(c.kappa_relativo(p[clave], q[clave], X0, X1) if X0 else None, 4),
                           c.fmt(I_ref, 8), c.fmt(I1, 8), c.fmt(c.kappa_relativo(p[clave], q[clave], I_ref, I1), 4)])
    tablas.append(c.tabla(
        "Sensibilidad de la amplitud permanente X y de I",
        ["Parámetro", "Perturbación rel.", "X [m]", "X' [m]", "κ_rel(X)", "I", "I'", "κ_rel(I)"],
        filas_sens,
        f"κ analítico de X respecto de Ω: {c.fmt(kappa_W, 5) if kappa_W is not None else '— (sin forzamiento)'}. Con F = 0 la amplitud permanente es nula.",
    ))

    # Indicadores.
    estatico = p["F0"] / p["k"] if p["F0"] > 0 else None
    indicadores = [
        c.indicador("ω₀ / ω_d", f"{c.fmt(an.w0, 5)} / {c.fmt(an.wd, 6)}", "rad/s"),
        c.indicador("Amplitud máxima (RK4)", resumen_edo["RK4"]["amp"], "m", f"referencia {c.fmt(max_ref, 6)} m"),
        c.indicador("Amplitud permanente X", an.X if p["F0"] > 0 else 0.0, "m", f"factor dinámico X/(F₀/k) = {c.fmt(an.X / estatico, 4)}" if estatico else "vibración libre"),
        c.indicador("I = ∫x² dt", I_ref, "m²·s", "referencia"),
        c.indicador("Error máx. Heun / RK4", f"{c.fmt(resumen_edo['Heun (RK2)']['error_max'], 3)} / {c.fmt(resumen_edo['RK4']['error_max'], 3)}", "m"),
        c.indicador("|R(hλ)| Heun / RK4", f"{c.fmt(resumen_edo['Heun (RK2)']['R'], 5)} / {c.fmt(resumen_edo['RK4']['R'], 5)}", "", "> 1: amplificación numérica"),
    ]
    if kappa_W is not None:
        indicadores.append(c.indicador("κ de X respecto de Ω", kappa_W, "", "sensibilidad del problema"))

    # Gráficos.
    t_d = np.linspace(0, T, 1500)
    graficos.append(c.grafico(
        "Desplazamiento de la cabina x(t)", "t [s]", "x [m]",
        [c.serie("Solución cerrada", t_d, an.x(t_d), color=0)]
        + [c.serie(met, r.t, r.y[:, 0], estilo="linea", color=i + 1, punteada=(met != "RK4")) for i, (met, r) in enumerate(sol.items())],
    ))
    graficos.append(c.grafico(
        "Error absoluto |x − x_ref| a lo largo del tiempo", "t [s]", "error [m]",
        [c.serie(met, r.t, np.abs(r.y[:, 0] - an.x(r.t)), color=i + 1) for i, (met, r) in enumerate(sol.items())],
        log_y=True,
    ))
    graficos.append(c.grafico(
        "Convergencia: error máximo frente al paso", "h [s]", "max |x − x_ref| [m]",
        [c.serie(met, [d[0] for d in datos], [d[1] for d in datos], estilo="linea-puntos", color=i + 1) for i, (met, datos) in enumerate(conv.items())],
        log_x=True, log_y=True, nota="Pendientes esperadas: 2 (Heun) y 4 (RK4) cuando el método es estable.",
    ))
    rs = np.linspace(0.5, 1.5, 400)
    curvas = []
    for i, z in enumerate(sorted({p["zeta"], 0.05, 0.005})):
        curvas.append(c.serie(f"ζ = {c.fmt(z)}", rs, [amplitud_permanente(dict(p, r=float(x), zeta=z, F0=1.0))[0] * p["k"] for x in rs], color=i + 1, punteada=(z != p["zeta"])))
    curvas.append(c.serie("Punto de operación", [p["r"]], [amplitud_permanente(dict(p, F0=1.0))[0] * p["k"]], estilo="puntos", color=0))
    graficos.append(c.grafico(
        "Curva de respuesta: factor dinámico X/(F₀/k)", "Ω/ω₀", "X/(F₀/k)", curvas, log_y=True,
        nota="Con ζ pequeño el pico es alto y estrecho: en su flanco, un pequeño cambio de Ω cambia mucho la amplitud.",
    ))
    graficos.append(c.grafico(
        "Integración de x(t)²: error frente a subintervalos", "n", "|I_n − I_ref|",
        [c.serie(nm, [d[0] for d in datos], [d[1] for d in datos], estilo="linea-puntos", color=i + 1) for i, (nm, datos) in enumerate(series_int.items())],
        log_x=True, log_y=True,
    ))

    # Conclusiones.
    eh, ek = resumen_edo["Heun (RK2)"], resumen_edo["RK4"]
    if tipo == "ESTRES":
        numerica = (f"Con h = {h} s (ω₀h = {c.fmt(an.w0 * h, 3)}) Heun tiene |R(hλ)| = {c.fmt(eh['R'], 4)} > 1: amplifica la oscilación en cada paso y su error máximo llega a "
                    f"{c.fmt(eh['error_max'], 3)} m, mientras RK4 (|R| = {c.fmt(ek['R'], 4)}) se queda en {c.fmt(ek['error_max'], 3)} m. Al reducir h, Heun vuelve a ser estable y ambos "
                    "recuperan sus órdenes 2 y 4. La diferencia Heun/RK4 es de estabilidad, no sólo de precisión.")
    elif tipo == "MAL_CONDICIONADO":
        numerica = (f"Con ζ = {p['zeta']} el problema es sensible: κ(X; Ω) = {c.fmt(kappa_W, 4)}, de modo que un error relativo de 0.1 % en Ω se convierte en ≈ {c.fmt(kappa_W * 0.1 if kappa_W else None, 3)} % en la amplitud. "
                    f"Numéricamente, el error de fase de Heun se acumula durante {c.fmt(T)} s (error máx. {c.fmt(eh['error_max'], 3)} m) mientras RK4 queda en {c.fmt(ek['error_max'], 3)} m; "
                    "ambos convergen al refinar h. La sensibilidad es del modelo, no un fallo de los métodos.")
    else:
        numerica = (f"En vibración libre RK4 reproduce la solución cerrada con error máximo {c.fmt(ek['error_max'], 3)} m y Heun con {c.fmt(eh['error_max'], 3)} m; "
                    "los órdenes observados se acercan a 4 y 2. Simpson y Romberg integran x² con muchas menos evaluaciones que el trapecio para el mismo error.")
    if p["F0"] > 0:
        problema = (f"La ráfaga con Ω/ω₀ = {c.fmt(p['r'])} produce una amplitud permanente de {c.fmt(an.X, 4)} m, {c.fmt(an.X / estatico, 4)} veces la deflexión estática F₀/k. "
                    "Cerca de la resonancia la amplitud depende sobre todo del amortiguamiento y de la frecuencia de la ráfaga, no de la masa por sí sola.")
        if an.X > 0.5:
            problema += " Amplitudes de ese orden superan el rango de validez de un modelo lineal de pequeñas oscilaciones."
    else:
        problema = (f"Sin viento, la oscilación inicial de {c.fmt(p['x0'])} m decae con envolvente e^(−ζω₀t): a los {c.fmt(T)} s queda "
                    f"{c.fmt(abs(p['x0']) * math.exp(-an.sigma * T), 3)} m. El amortiguamiento controla cuánto dura el balanceo.")
    limites = "Modelo lineal de un grado de libertad con viento armónico idealizado; no incluye el balanceo pendular real, la turbulencia ni el acoplamiento con el cable."

    return {
        "descripcion": ESCENARIOS[tipo]["descripcion"],
        "indicadores": indicadores,
        "ejecuciones": ejecuciones,
        "tablas": tablas,
        "graficos": graficos,
        "conclusiones": {"numerica": numerica, "problema": problema, "limites": limites},
        "panel": [
            ["Amplitud máx.", f"{c.fmt(max_ref, 4)} m"],
            ["Error máx. Heun / RK4", f"{c.fmt(eh['error_max'], 2)} / {c.fmt(ek['error_max'], 2)} m"],
            ["κ(X; Ω)", c.fmt(kappa_W, 4) if kappa_W is not None else "— (F = 0)"],
        ],
    }
