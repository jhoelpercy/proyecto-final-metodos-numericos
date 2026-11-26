"""Módulo 1: detención de emergencia de cabinas.

Modelo de la fase de frenado (la reacción se recorre a velocidad constante):
    dx/dt = v,   dv/dt = −a(v, t),   x(0) = 0,   v(0) = v0,   d_total = v0 t_r + x(t_det).
Escenarios:
    NORMAL            a = a0 constante              → v lineal, x cuadrática (todos los métodos son exactos).
    ESTRES            a = a0 + c2 v² (resistencia)  → v(t) = s tan(φ − k t), solución cerrada.
    MAL_CONDICIONADO  v(t) = v0 (1 − t/T)^5         → raíz de multiplicidad 5 en t = T (subexperimento numérico).
"""

from __future__ import annotations

import math

import numpy as np

from ..numerico import edo, integracion, raices
from . import comun as c

CODIGO = "detencion"
ID = 1
NOMBRE = "Detención de emergencia"

PARAMETROS = [
    {"clave": "v0", "etiqueta": "Velocidad inicial v₀", "unidad": "m/s", "min": 0.5, "max": 30, "paso": 0.1},
    {"clave": "a0", "etiqueta": "Desaceleración base a₀", "unidad": "m/s²", "min": 0.1, "max": 10, "paso": 0.1, "escenarios": ["NORMAL", "ESTRES"]},
    {"clave": "c2", "etiqueta": "Coef. resistencia c₂ (a = a₀ + c₂v²)", "unidad": "1/m", "min": 0, "max": 1, "paso": 0.01, "escenarios": ["NORMAL", "ESTRES"]},
    {"clave": "T", "etiqueta": "Duración del frenado plano T", "unidad": "s", "min": 0.5, "max": 30, "paso": 0.1, "escenarios": ["MAL_CONDICIONADO"]},
    {"clave": "tr", "etiqueta": "Tiempo de reacción t_r", "unidad": "s", "min": 0, "max": 5, "paso": 0.1},
    {"clave": "h", "etiqueta": "Paso de la EDO h", "unidad": "s", "min": 0.001, "max": 2, "paso": 0.01},
    {"clave": "tol", "etiqueta": "Tolerancia de raíces", "unidad": "", "min": 1e-15, "max": 1e-2, "paso": "any"},
    {"clave": "d_disp", "etiqueta": "Distancia disponible", "unidad": "m", "min": 1, "max": 500, "paso": 0.5},
]

ESCENARIOS = {
    "NORMAL": {
        "descripcion": "Frenado con desaceleración constante: caso analítico de verificación.",
        "parametros": {"v0": 5.0, "a0": 2.5, "c2": 0.0, "tr": 0.5, "h": 0.1, "tol": 1e-10, "d_disp": 10.0},
        "fuente": "Sintético: valores nominales supuestos (orden de magnitud de una cabina monocable).",
    },
    "ESTRES": {
        "descripcion": "Mayor velocidad y reacción, frenado débil con resistencia cuadrática y paso grande h = 0.5 s.",
        "parametros": {"v0": 7.0, "a0": 1.2, "c2": 0.05, "tr": 1.5, "h": 0.5, "tol": 1e-13, "d_disp": 20.0},
        "fuente": "Sintético: condición exigente dentro del mismo modelo.",
    },
    "MAL_CONDICIONADO": {
        "descripcion": "Subexperimento numérico: frenado plano v = v₀(1 − t/T)⁵, raíz múltiple con v′(T) = 0.",
        "parametros": {"v0": 5.0, "T": 4.0, "tr": 0.5, "h": 0.1, "tol": 1e-10, "d_disp": 15.0},
        "fuente": "Sintético: perfil elegido para que el instante de detención sea una raíz de multiplicidad 5.",
    },
}

PERTURBACION_SENSOR = 1e-12  # desplazamiento de v en m/s para medir la sensibilidad de la raíz


class Modelo:
    """Solución de referencia (cerrada) y campo de la EDO para cada escenario."""

    def __init__(self, tipo: str, p: dict):
        self.tipo, self.p = tipo, p
        self.v0 = p["v0"]
        if tipo == "MAL_CONDICIONADO":
            self.T = p["T"]
            self.t_det = self.T
            self.d_fren = self.v0 * self.T / 6.0
            self.formula = "v(t) = v₀(1 − t/T)⁵"
        else:
            self.a0, self.c2 = p["a0"], p.get("c2", 0.0)
            if self.c2 == 0:
                self.t_det = self.v0 / self.a0
                self.d_fren = self.v0**2 / (2 * self.a0)
                self.formula = "v(t) = v₀ − a₀t"
            else:
                self.k = math.sqrt(self.a0 * self.c2)
                self.s = math.sqrt(self.a0 / self.c2)
                self.phi = math.atan(self.v0 / self.s)
                self.t_det = self.phi / self.k
                self.d_fren = math.log1p(self.c2 * self.v0**2 / self.a0) / (2 * self.c2)
                self.formula = "v(t) = s·tan(φ − k t), s = √(a₀/c₂), k = √(a₀c₂)"
        self.d_total = self.v0 * p["tr"] + self.d_fren

    def v(self, t: float) -> float:
        if self.tipo == "MAL_CONDICIONADO":
            return self.v0 * (1 - t / self.T) ** 5
        if self.c2 == 0:
            return self.v0 - self.a0 * t
        return self.s * math.tan(self.phi - self.k * t)

    def dv(self, t: float) -> float:
        if self.tipo == "MAL_CONDICIONADO":
            return -5 * self.v0 / self.T * (1 - t / self.T) ** 4
        return -(self.a0 + self.c2 * self.v(t) ** 2)

    def x(self, t: float) -> float:
        if self.tipo == "MAL_CONDICIONADO":
            return self.v0 * self.T / 6.0 * (1 - (1 - t / self.T) ** 6)
        if self.c2 == 0:
            return self.v0 * t - 0.5 * self.a0 * t * t
        return math.log(math.cos(self.phi - self.k * t) / math.cos(self.phi)) / self.c2

    def campo(self, t: float, y: np.ndarray) -> np.ndarray:
        if self.tipo == "MAL_CONDICIONADO":
            return np.array([y[1], -5 * self.v0 / self.T * (1 - t / self.T) ** 4])
        return np.array([y[1], -(self.a0 + self.c2 * y[1] ** 2)])


def _buscador(tol: float):
    """Localiza el último paso h* con secante; si falla, bisección (siempre hay cambio de signo)."""

    def buscar(g, a, b):
        r = raices.secante(g, a, b, tol=tol, max_iter=200)
        if not r.convergio or not (a <= r.raiz <= b):
            r = raices.biseccion(g, a, b, tol=tol, max_iter=500)
        return r

    return buscar


def _resolver_edo(modelo: Modelo, metodo: str, h: float, tol: float):
    return edo.integrar_hasta_evento(modelo.campo, [0.0, modelo.v0], h, lambda y: y[1], metodo, t_max=1e4, buscador=_buscador(tol))


def _intervalo(f, paso: float = 0.3, limite: int = 2000) -> tuple[float, float]:
    """Búsqueda incremental desde t = 0 hasta encontrar cambio de signo de f."""
    a, fa = 0.0, f(0.0)
    for i in range(1, limite + 1):
        b = i * paso
        fb = f(b)
        if fb == 0.0:
            return a, b + paso
        if fa * fb < 0:
            return a, b
        a, fa = b, fb
    raise ValueError("No se encontró un intervalo con cambio de signo.")


def _distancia_total(tipo: str, p: dict) -> float:
    return Modelo(tipo, p).d_total


def ejecutar(tipo: str, p: dict) -> dict:
    m = Modelo(tipo, p)
    h, tol = p["h"], p["tol"]
    ejecuciones, tablas, graficos, indicadores = [], [], [], []

    # 1) EDO: Heun y RK4 con evento v = 0 en el último paso.
    sol = {}
    for metodo in ("Heun (RK2)", "RK4"):
        (r, raiz_paso), ms = c.cronometrar(lambda: _resolver_edo(m, metodo, h, tol))
        sol[metodo] = r
        filas = []
        for i, (ti, (xi, vi)) in enumerate(zip(r.t, r.y)):
            xr, vr = m.x(ti), m.v(ti)
            ea, er = c.errores(xr, xi)
            filas.append(c.fila(i, {"t": ti, "x": xi, "v": vi, "x_ref": xr, "v_ref": vr, "error_v": abs(vi - vr)}, None, ea, er))
        t_det, d_fren = float(r.t[-1]), float(r.y[-1, 0])
        e_t, er_t = c.errores(m.t_det, t_det)
        e_d, er_d = c.errores(m.d_fren, d_fren)
        ejecuciones.append(
            c.ejecucion(
                metodo, "EDO", filas,
                referencia=f"solución cerrada {m.formula}; t_det = {c.fmt(m.t_det)} s, d_fren = {c.fmt(m.d_fren)} m",
                valor={"t_det": t_det, "d_fren": d_fren, "d_total": m.v0 * p["tr"] + d_fren, "pasos": r.pasos, "ultimo_paso": raiz_paso.raiz if raiz_paso else None},
                metricas={"error_abs_t_det": e_t, "error_rel_t_det": er_t, "error_abs_d_fren": e_d, "error_rel_d_fren": er_d, "error_max_x": max((f["error_abs"] or 0) for f in filas)},
                interpretacion=f"{metodo} con h = {h} s: se detiene en t = {c.fmt(t_det)} s tras {c.fmt(d_fren)} m de frenado (error en t_det {c.fmt(e_t)} s, en distancia {c.fmt(e_d)} m).",
                convergio=r.estable, criterio=r.criterio + (f"; h* = {c.fmt(raiz_paso.raiz)} s por secante" if raiz_paso else ""),
                tiempo_ms=ms, tolerancia=tol, paso=h, total=r.pasos,
            )
        )

    # 2) Convergencia de la EDO al refinar el paso.
    pasos = [h, h / 2, h / 4, h / 8]
    conv = {met: [] for met in sol}
    for met in conv:
        for hh in pasos:
            r, _ = _resolver_edo(m, met, hh, tol)
            conv[met].append((hh, abs(r.t[-1] - m.t_det), abs(r.y[-1, 0] - m.d_fren)))
    filas_conv = []
    for met, datos in conv.items():
        for i, (hh, et, ed) in enumerate(datos):
            p_t = c.orden(datos[i - 1][1], et) if i else None
            p_d = c.orden(datos[i - 1][2], ed) if i else None
            filas_conv.append([met, c.fmt(hh), c.fmt(et), c.fmt(p_t, 3), c.fmt(ed), c.fmt(p_d, 3)])
    tablas.append(c.tabla(
        "Refinamiento del paso de la EDO (error frente a la solución cerrada)",
        ["Método", "h [s]", "|t_det − t*| [s]", "orden p (t)", "|d − d*| [m]", "orden p (d)"],
        filas_conv,
        "p = ln(e_h / e_(h/2)) / ln 2. Se omite cuando el error está en el piso del redondeo (< 1e−13).",
    ))

    # 3) Raíces de v(t) = 0 sobre la formulación continua del modelo.
    a, b = _intervalo(m.v)
    corridas = [
        ("Bisección", lambda: raices.biseccion(m.v, a, b, tol, 500)),
        ("Newton-Raphson", lambda: raices.newton_raphson(m.v, m.dv, 0.0, tol, 500)),
        ("Secante", lambda: raices.secante(m.v, 0.0, 0.3, tol, 500)),
    ]
    resultados_raiz = {}
    for nombre, correr in corridas:
        r, ms = c.cronometrar(correr)
        resultados_raiz[nombre] = r
        ejecuciones.append(c.ejecucion_raiz(nombre, r, m.t_det, m.v, tol=tol, max_iter=500, tiempo_ms=ms, contexto="Instante de detención v(t) = 0", unidad="s"))
    tablas.append(c.tabla(
        "Raíces de v(t) = 0",
        ["Método", "Semillas / intervalo", "t_det [s]", "Iteraciones", "|t − t*| [s]", "|v(t)| residual", "Orden observado", "Criterio"],
        [
            [n, f"[{c.fmt(a)}, {c.fmt(b)}]" if n == "Bisección" else ("t₀ = 0" if n == "Newton-Raphson" else "t₀ = 0, t₁ = 0.3"),
             c.fmt(r.raiz, 12), r.iteraciones, c.fmt(abs(r.raiz - m.t_det)), c.fmt(abs(m.v(r.raiz))),
             c.fmt(raices.orden_convergencia_estimado([h_["x"] for h_ in r.historial], m.t_det), 3), r.criterio]
            for n, r in resultados_raiz.items()
        ],
        "Intervalo de bisección obtenido por búsqueda incremental con paso 0.3 s.",
    ))

    # Sensibilidad de la raíz ante un desplazamiento de δ en la velocidad medida.
    d = PERTURBACION_SENSOR
    a2, b2 = _intervalo(lambda t: m.v(t) - d)
    t_pert = raices.biseccion(lambda t: m.v(t) - d, a2, b2, 1e-15, 500).raiz
    t_base = raices.biseccion(m.v, a, b, 1e-15, 500).raiz
    amplificacion = abs(t_pert - t_base) / d

    # 4) Integración de v(t) en [0, t_det] = distancia de frenado.
    t_lim = resultados_raiz["Bisección"].raiz
    filas_int = []
    serie_trap, serie_simp, valores_int = [], [], {}
    previo = {}
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        vt = integracion.trapecio(m.v, 0.0, t_lim, n)
        vs = integracion.simpson(m.v, 0.0, t_lim, n)
        et, es = abs(vt - m.d_fren), abs(vs - m.d_fren)
        filas_int.append([n, c.fmt(t_lim / n), c.fmt(vt, 12), c.fmt(et), c.fmt(c.orden(previo.get("t"), et), 3), c.fmt(vs, 12), c.fmt(es), c.fmt(c.orden(previo.get("s"), es), 3)])
        previo = {"t": et, "s": es}
        serie_trap.append((n, et))
        serie_simp.append((n, es))
        valores_int[n] = (vt, vs)
    tablas.append(c.tabla(
        "Refinamiento de ∫₀^t_det v(t) dt (distancia de frenado)",
        ["n", "h [s]", "Trapecio [m]", "Error", "orden", "Simpson 1/3 [m]", "Error", "orden"],
        filas_int,
        f"Referencia analítica d_fren = {c.fmt(m.d_fren, 12)} m. Límite superior: raíz por bisección.",
    ))
    rom, ms_rom = c.cronometrar(lambda: integracion.romberg(m.v, 0.0, t_lim, tol=max(tol, 1e-14), max_niveles=12))
    tablas.append(c.tabla(
        "Tabla de Romberg R(i, j)",
        ["i", "n = 2^i"] + [f"R(i,{j})" for j in range(len(rom.tabla))],
        [[i, 2**i] + [c.fmt(v, 12) for v in fila_] + [""] * (len(rom.tabla) - len(fila_)) for i, fila_ in enumerate(rom.tabla)],
        f"Criterio: {rom.criterio}. Evaluaciones de v(t): {rom.evaluaciones}.",
    ))
    for j, (nombre, filas_metodo) in enumerate([("Trapecio compuesto", serie_trap), ("Simpson 1/3 compuesto", serie_simp)]):
        valor = valores_int[256][j]
        ea, er = c.errores(m.d_fren, valor)
        ejecuciones.append(c.ejecucion(
            nombre, "Integración",
            [c.fila(i + 1, {"n": n, "valor": valores_int[n][j]}, None, e, e / m.d_fren) for i, (n, e) in enumerate(filas_metodo)],
            referencia=f"d_fren analítica = {c.fmt(m.d_fren, 12)} m", valor={"d_fren": valor, "n": 256},
            metricas={"error_abs": ea, "error_rel": er}, interpretacion=f"{nombre} con n = 256: {c.fmt(valor, 10)} m (error {c.fmt(ea)} m).",
            paso=t_lim / 256,
        ))
    ea, er = c.errores(m.d_fren, rom.valor)
    ejecuciones.append(c.ejecucion(
        "Romberg", "Integración",
        [c.fila(i, {"n": 2**i, "diagonal": fila_[-1]}, None, *c.errores(m.d_fren, fila_[-1])) for i, fila_ in enumerate(rom.tabla)],
        referencia=f"d_fren analítica = {c.fmt(m.d_fren, 12)} m", valor={"d_fren": rom.valor, "niveles": rom.niveles},
        metricas={"error_abs": ea, "error_rel": er, "evaluaciones": rom.evaluaciones},
        interpretacion=f"Romberg alcanza {c.fmt(rom.valor, 12)} m con {rom.evaluaciones} evaluaciones (error {c.fmt(ea)} m).",
        convergio=rom.convergio, criterio=rom.criterio, tiempo_ms=ms_rom, tolerancia=max(tol, 1e-14), max_iter=12,
    ))

    # 5) Sensibilidad física de la distancia total (perturbación +1 %).
    claves = ["v0", "tr"] + (["T"] if tipo == "MAL_CONDICIONADO" else ["a0"] + (["c2"] if p.get("c2", 0) > 0 else []))
    filas_sens = []
    for k in claves:
        q = dict(p)
        if q[k] == 0:
            continue
        q[k] = p[k] * 1.01
        d1 = _distancia_total(tipo, q)
        filas_sens.append([k, c.fmt(p[k]), c.fmt(q[k]), c.fmt(m.d_total, 8), c.fmt(d1, 8), c.fmt(d1 - m.d_total), c.fmt(c.kappa_relativo(p[k], q[k], m.d_total, d1), 4)])
    tablas.append(c.tabla(
        "Sensibilidad de la distancia total ante +1 % en cada parámetro",
        ["Parámetro", "Valor", "Perturbado", "d_total [m]", "d_total' [m]", "Δd [m]", "κ_rel"],
        filas_sens,
        "κ_rel = |Δd/d| / |Δp/p|. En v₀ queda entre 1 (la reacción es lineal en v₀) y 2 (el frenado es cuadrático en v₀).",
    ))

    # Indicadores.
    rk4 = ejecuciones[1]["resumen"]["valor"]
    margen = p["d_disp"] - rk4["d_total"]
    indicadores += [
        c.indicador("Tiempo de detención (RK4)", rk4["t_det"], "s", f"referencia {c.fmt(m.t_det)} s"),
        c.indicador("Distancia de frenado (RK4)", rk4["d_fren"], "m", f"referencia {c.fmt(m.d_fren)} m"),
        c.indicador("Distancia total", rk4["d_total"], "m", f"reacción {c.fmt(m.v0 * p['tr'])} m"),
        c.indicador("Margen frente a la distancia disponible", margen, "m", "negativo: no se detiene a tiempo"),
        c.indicador("Amplificación de δv en t_det", amplificacion, "s/(m/s)", f"δ = {PERTURBACION_SENSOR:g} m/s"),
        c.indicador("Iteraciones bisección / Newton / secante", " / ".join(str(r.iteraciones) for r in resultados_raiz.values())),
    ]

    # Gráficos.
    malla = np.linspace(0, max(m.t_det, sol["RK4"].t[-1], sol["Heun (RK2)"].t[-1]), 200)
    graficos.append(c.grafico(
        "Velocidad durante el frenado", "t [s]", "v [m/s]",
        [
            c.serie("Solución cerrada", malla, [m.v(t) if t <= m.t_det else 0.0 for t in malla], color=0),
            c.serie("Heun (RK2)", sol["Heun (RK2)"].t, sol["Heun (RK2)"].y[:, 1], estilo="puntos", color=1),
            c.serie("RK4", sol["RK4"].t, sol["RK4"].y[:, 1], estilo="puntos", color=2),
        ],
        nota="Los puntos son los pasos de la EDO; el último paso está recortado al evento v = 0.",
    ))
    graficos.append(c.grafico(
        "Convergencia de la EDO: error en la distancia de frenado", "h [s]", "|d − d*| [m]",
        [c.serie(met, [q[0] for q in datos], [q[2] for q in datos], estilo="linea-puntos", color=i + 1) for i, (met, datos) in enumerate(conv.items())],
        log_x=True, log_y=True,
        nota="Pendiente 2 para Heun y 4 para RK4 en escala log-log; si no aparece una serie, su error es 0 (exacto).",
    ))
    graficos.append(c.grafico(
        "Convergencia de las raíces de v(t) = 0", "iteración k", "|t_k − t*| [s]",
        [c.serie(n, [h_["k"] for h_ in r.historial], [abs(h_["x"] - m.t_det) for h_ in r.historial], estilo="linea-puntos", color=i + 1) for i, (n, r) in enumerate(resultados_raiz.items())],
        log_y=True,
    ))
    graficos.append(c.grafico(
        "Integración de v(t): error frente a subintervalos", "n", "error [m]",
        [c.serie("Trapecio", *zip(*serie_trap), estilo="linea-puntos", color=1), c.serie("Simpson 1/3", *zip(*serie_simp), estilo="linea-puntos", color=2)],
        log_x=True, log_y=True,
    ))
    v_malla = np.linspace(0.5 * m.v0, 1.6 * m.v0, 40)
    comparacion = []
    for i, extra in enumerate([0.0, 1.0]):
        q = dict(p, tr=p["tr"] + extra)
        comparacion.append(c.serie(f"t_r = {c.fmt(q['tr'])} s", v_malla, [_distancia_total(tipo, dict(q, v0=float(v))) for v in v_malla], color=i + 1))
    comparacion.append(c.serie("Distancia disponible", [v_malla[0], v_malla[-1]], [p["d_disp"]] * 2, color=0, punteada=True))
    graficos.append(c.grafico("Comparación: distancia total según velocidad y reacción", "v₀ [m/s]", "d_total [m]", comparacion, nota="Curvas del modelo cerrado con los demás parámetros fijos."))

    # Conclusiones.
    e_heun = ejecuciones[0]["resumen"]["metricas"]["error_abs_d_fren"]
    e_rk4 = ejecuciones[1]["resumen"]["metricas"]["error_abs_d_fren"]
    it = {n: r.iteraciones for n, r in resultados_raiz.items()}
    if tipo == "NORMAL":
        numerica = (f"Con desaceleración constante v(t) es lineal y x(t) cuadrática: Heun y RK4 reproducen la solución hasta el redondeo "
                    f"(error en d: {c.fmt(e_heun)} y {c.fmt(e_rk4)} m), trapecio y Simpson son exactos y Newton y secante terminan en "
                    f"{it['Newton-Raphson']} y {it['Secante']} iteraciones, frente a {it['Bisección']} de la bisección. Es un caso de verificación, no de comparación de órdenes.")
    elif tipo == "ESTRES":
        numerica = (f"Con resistencia cuadrática y h = {h} s, Heun comete {c.fmt(e_heun)} m de error en la distancia y RK4 {c.fmt(e_rk4)} m; "
                    "al refinar el paso los órdenes observados se acercan a 2 y 4. Newton converge cuadráticamente "
                    f"({it['Newton-Raphson']} iteraciones), la secante superlinealmente ({it['Secante']}) y la bisección linealmente ({it['Bisección']}). "
                    "Romberg alcanza la tolerancia con muchas menos evaluaciones que el trapecio.")
    else:
        numerica = (f"La raíz t = T tiene multiplicidad 5: Newton pierde la convergencia cuadrática y avanza con razón 4/5 ({it['Newton-Raphson']} iteraciones), "
                    f"la secante también se vuelve lineal ({it['Secante']}) y el residuo |v(t)| es muchísimo menor que el error en t. "
                    f"Un desplazamiento de {PERTURBACION_SENSOR:g} m/s en v mueve la raíz {c.fmt(abs(t_pert - t_base))} s (amplificación {c.fmt(amplificacion)}): "
                    "es mal condicionamiento del problema, no fallo de convergencia, porque los tres métodos sí convergen. "
                    f"El t_det de las EDO tiene un error de {c.fmt(ejecuciones[1]['resumen']['metricas']['error_abs_t_det'])} s con RK4 aunque la distancia es precisa.")
    if margen >= 0:
        problema = f"La cabina se detiene en {c.fmt(rk4['d_total'], 4)} m, con un margen de {c.fmt(margen, 3)} m respecto a los {c.fmt(p['d_disp'])} m disponibles."
    else:
        problema = f"La cabina necesita {c.fmt(rk4['d_total'], 4)} m y excede en {c.fmt(-margen, 3)} m la distancia disponible de {c.fmt(p['d_disp'])} m."
    kappas = {f[0]: f[6] for f in filas_sens}
    problema += f" De ese total, {c.fmt(m.v0 * p['tr'], 3)} m se recorren durante la reacción."
    try:
        kv, kt = float(kappas.get("v0")), float(kappas.get("tr"))
        mayor = "v₀" if kv >= kt else "t_r"
        problema += f" La distancia total es más sensible a {mayor}: κ_rel = {kappas['v0']} para v₀ y {kappas['tr']} para t_r."
    except (TypeError, ValueError):
        pass
    if tipo == "MAL_CONDICIONADO":
        problema += " En el frenado plano el instante exacto de detención está mal definido (la cabina «se arrastra» al final), pero la distancia sí es confiable."
    limites = "Masa puntual en trayectoria rectilínea; no modela pendiente del cable, elasticidad ni el sistema real de frenos. Datos sintéticos."

    return {
        "descripcion": ESCENARIOS[tipo]["descripcion"],
        "indicadores": indicadores,
        "ejecuciones": ejecuciones,
        "tablas": tablas,
        "graficos": graficos,
        "conclusiones": {"numerica": numerica, "problema": problema, "limites": limites},
        "panel": [
            ["d_total", c.fmt(rk4["d_total"], 5) + " m"],
            ["Error RK4 / Heun en d", f"{c.fmt(e_rk4, 2)} / {c.fmt(e_heun, 2)} m"],
            ["Iter. bisección / Newton / secante", " / ".join(str(v) for v in it.values())],
        ],
    }
