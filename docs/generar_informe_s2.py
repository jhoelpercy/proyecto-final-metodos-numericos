"""Genera las figuras y tablas del informe de la semana 3 a partir de los resultados reales.

Uso, desde la raíz del repositorio:
    python docs/generar_informe_s2.py
Escribe docs/fig/s2_*.png y docs/informe_s2/*.tex; luego se compila docs/informe_semana3.tex con XeLaTeX.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from backend.simtele import persistencia  # noqa: E402
from backend.simtele.casos import CASOS, comun, simular  # noqa: E402

FIG = RAIZ / "docs" / "fig"
TAB = RAIZ / "docs" / "informe_s2"
COLORES = ["#14222e", "#2a78d6", "#eb6834", "#1baf7a", "#7c3aed", "#eda100"]
NOMBRE_TIPO = {"NORMAL": "NORMAL", "ESTRES": "ESTRÉS", "MAL_CONDICIONADO": "MAL COND."}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6ebef", "legend.fontsize": 7.5, "figure.dpi": 150})


# ---------- figuras ----------
def dibujar(ax, g, titulo=None):
    for s in g["series"]:
        if not s["puntos"]:
            continue
        x, y = zip(*s["puntos"])
        color = COLORES[s["color"] if s["color"] is not None else 1]
        estilo = "--" if s["punteada"] else "-"
        if s["estilo"] == "puntos":
            ax.plot(x, y, "o", ms=3.2, color=color, label=s["nombre"], zorder=3)
        elif s["estilo"] == "linea-puntos":
            ax.plot(x, y, estilo, marker="o", ms=3, lw=1.4, color=color, label=s["nombre"])
        else:
            ax.plot(x, y, estilo, lw=1.4, color=color, label=s["nombre"])
    if g["log_x"]:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if g["log_y"]:
        ax.set_yscale("log")
    ax.set_xlabel(g["eje_x"])
    ax.set_ylabel(g["eje_y"])
    ax.set_title(titulo or g["titulo"], fontsize=9)
    ax.legend(frameon=True, framealpha=0.85, edgecolor="none", fontsize=6.8)


def figura(nombre, paneles, alto=3.0):
    fig, ejes = plt.subplots(1, len(paneles), figsize=(3.6 * len(paneles), alto))
    for ax, (r, indice, titulo) in zip(ejes, paneles):
        g = next(x for x in r["graficos"] if x["titulo"].startswith(indice)) if isinstance(indice, str) else r["graficos"][indice]
        dibujar(ax, g, f"{NOMBRE_TIPO[r['tipo']]}: {titulo}")
    fig.tight_layout()
    fig.savefig(FIG / nombre, dpi=200)
    plt.close(fig)


def figura_der():
    """DER v2: mismas 6 entidades; se resaltan los campos nuevos."""
    entidades = {
        "caso_estudio": (0.2, 6.6, ["id PK", "nombre UK", "modelo_version"]),
        "escenario": (4.5, 6.6, ["id PK", "caso_id FK", "tipo", "parametros_json", "fuente_datos"]),
        "simulacion": (8.8, 6.6, ["id PK", "escenario_id FK", "fecha", "usuario_equipo", "estado", "+ resultado_json", "+ tiempo_total_ms"]),
        "ejecucion_metodo": (4.5, 2.7, ["id PK", "simulacion_id FK", "metodo", "tolerancia, paso, max_iter", "convergio, tiempo_ms", "+ familia, criterio, orden"]),
        "iteracion_resultado": (0.2, 2.7, ["id PK", "ejecucion_id FK", "numero", "aproximacion_json", "residual", "error_abs, error_rel"]),
        "resultado_resumen": (8.8, 2.7, ["id PK", "ejecucion_id FK UK", "referencia", "valor_json", "metrica_error_json", "interpretacion"]),
    }
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.7, 6.8)
    ax.axis("off")
    cajas = {}
    for nombre, (x, y, campos) in entidades.items():
        alto = 0.42 * (len(campos) + 1)
        ax.add_patch(FancyBboxPatch((x, y - alto), 3.0, alto, boxstyle="round,pad=0.02", fc="white", ec="#1f6fbf", lw=1.2))
        ax.add_patch(FancyBboxPatch((x, y - 0.42), 3.0, 0.42, boxstyle="round,pad=0.02", fc="#1f6fbf", ec="#1f6fbf"))
        ax.text(x + 1.5, y - 0.21, nombre, ha="center", va="center", color="white", weight="bold", fontsize=9)
        for i, c in enumerate(campos):
            ax.text(x + 0.12, y - 0.42 * (i + 1.5), c, va="center", fontsize=8, color="#b45309" if c.startswith("+") else "#14222e", weight="bold" if c.startswith("+") else "normal")
        cajas[nombre] = (x, y, alto)

    def flecha(a, b, texto, lado_a="r", lado_b="l"):
        xa, ya, ha = cajas[a]
        xb, yb, hb = cajas[b]
        pa = (xa + 3.0, ya - 0.63) if lado_a == "r" else ((xa, ya - 0.63) if lado_a == "l" else (xa + 1.5, ya - ha))
        pb = (xb, yb - 0.63) if lado_b == "l" else ((xb + 3.0, yb - 0.63) if lado_b == "r" else (xb + 1.5, yb))
        ax.annotate("", xy=pb, xytext=pa, arrowprops=dict(arrowstyle="-|>", color="#4f6070", lw=1))
        ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 0.22, texto, ha="center", fontsize=7.5, color="#4f6070")

    flecha("caso_estudio", "escenario", "1 : N  se prueba en")
    flecha("escenario", "simulacion", "1 : N  se ejecuta")
    flecha("simulacion", "ejecucion_metodo", "1 : N  contiene", "b", "t")
    flecha("ejecucion_metodo", "iteracion_resultado", "1 : N  registra", "l", "r")
    flecha("ejecucion_metodo", "resultado_resumen", "1 : 1  resume", "r", "l")
    ax.text(0.2, -0.55, "En naranja: campos agregados en la semana 3.", fontsize=8, color="#b45309")
    fig.tight_layout()
    fig.savefig(FIG / "s2_der.png", dpi=200)
    plt.close(fig)


# ---------- LaTeX ----------
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SIMBOLOS = {
    "κ": r"\kappa", "ρ": r"\rho", "λ": r"\lambda", "Λ": r"\Lambda", "ζ": r"\zeta", "ω": r"\omega", "Ω": r"\Omega", "φ": r"\varphi", "σ": r"\sigma",
    "δ": r"\delta", "Δ": r"\Delta", "α": r"\alpha", "β": r"\beta", "∫": r"\int", "√": r"\sqrt{}", "≈": r"\approx", "≤": r"\le", "≥": r"\ge", "−": "-",
    "×": r"\times", "·": r"\cdot", "‖": r"\|", "∞": r"\infty", "→": r"\to", "⇒": r"\Rightarrow", "′": "'", "″": "''", "±": r"\pm", "ξ": r"\xi", "ᵣ": "_r",
}


def tex(texto) -> str:
    s = str(texto)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    s = s.replace("✓", "sí").replace("✕", "no")
    s = re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "$^{" + m.group(0).translate(SUP) + "}$", s)
    s = re.sub("[₀₁₂₃₄₅₆₇₈₉]+", lambda m: "$_{" + m.group(0).translate(SUB) + "}$", s)
    s = "".join(f"${SIMBOLOS[c]}$" if c in SIMBOLOS else c for c in s)
    s = re.sub(r"(\d(?:\.\d+)?)e([-+]?\d+)", lambda m: f"{m.group(1)}$\\times10^{{{int(m.group(2))}}}$", s)
    return s.replace("$$", "")


def tabla_tex(columnas, filas, alinear=None, tam=r"\footnotesize") -> str:
    alinear = alinear or "l" * len(columnas)
    cuerpo = "\n".join(" & ".join(tex(v) for v in f) + r" \\" for f in filas)
    return (f"{{{tam}\n\\begin{{longtable}}{{{alinear}}}\n\\toprule\n" + " & ".join(rf"\textbf{{{tex(c)}}}" for c in columnas)
            + r" \\" + "\n\\midrule\n\\endhead\n" + cuerpo + "\n\\bottomrule\n\\end{longtable}\n}\n")


def tabla_spec(r, prefijo, columnas=None, max_filas=None) -> str:
    t = next(x for x in r["tablas"] if x["titulo"].startswith(prefijo))
    indices = [t["columnas"].index(c) for c in columnas] if columnas else list(range(len(t["columnas"])))
    filas = [[f[i] for i in indices] for f in t["filas"][:max_filas]]
    return tabla_tex([t["columnas"][i] for i in indices], filas, "l" + "r" * (len(indices) - 1))


ELEGIR = {
    ("EDO", "detencion"): ("d_fren", "error_abs_d_fren", "error_rel_d_fren"),
    ("Integración", "detencion"): ("d_fren", "error_abs", "error_rel"),
    ("EDO", "oscilaciones"): ("amplitud_max", "error_max_x", None),
    ("Integración", "oscilaciones"): ("I_muestras_rk4", "error_abs_muestras_rk4", "error_rel_muestras_rk4"),
    ("Interpolación", "sensores"): ("error_max_oculto", "error_max_oculto", "error_max_malla"),
}


def tabla_errores(codigo, resultados) -> str:
    filas = []
    for tipo in ("NORMAL", "ESTRES", "MAL_CONDICIONADO"):
        r = resultados[(codigo, tipo)]
        for e in r["ejecuciones"]:
            v, m = e["resumen"]["valor"], e["resumen"]["metricas"]
            clave = ELEGIR.get((e["familia"], codigo))
            if e["familia"] == "Raíces":
                valor, ea, er = v.get("raiz"), m.get("error_abs"), m.get("error_rel")
                tercera = f"res. {comun.fmt(m.get('residual'), 3)}"
            elif e["familia"] == "Sistemas lineales":
                valor, ea, er = None, m.get("error_abs"), m.get("error_rel")
                tercera = f"res. {comun.fmt(m.get('residual'), 3)}"
            elif clave:
                fuente = lambda k: (m.get(k) if k in m else v.get(k)) if k else None
                valor, ea = fuente(clave[0]), fuente(clave[1])
                if e["etiqueta"] == "Romberg" and codigo == "oscilaciones":
                    valor, ea = v.get("I_analitica"), m.get("error_abs")
                er = fuente(clave[2]) if clave[2] else None
                tercera = comun.fmt(er, 3)
                if e["familia"] == "Interpolación":
                    valor, tercera = m.get("rms_oculto"), f"malla {comun.fmt(m.get('error_max_malla'), 3)}"
            else:
                valor, ea, tercera = None, None, ""
            filas.append([NOMBRE_TIPO[tipo], e["familia"], e["etiqueta"], e["total_iteraciones"], "sí" if e["convergio"] else "no",
                          comun.fmt(valor, 8), comun.fmt(ea, 3), tercera])
    return tabla_tex(["Escenario", "Familia", "Método", "Iter./pasos", "Conv.", "Resultado", "Error abs.", "Error rel. / otro"], filas, "lllrlrrr")


def conclusiones_tex(codigo, resultados) -> str:
    partes = [r"\begin{itemize}"]
    for tipo in ("NORMAL", "ESTRES", "MAL_CONDICIONADO"):
        c = resultados[(codigo, tipo)]["conclusiones"]
        partes.append(rf"\item \textbf{{{NOMBRE_TIPO[tipo]}.}} \emph{{Numérica:}} {tex(c['numerica'])} \emph{{Problema:}} {tex(c['problema'])}")
    partes.append(r"\end{itemize}")
    return "\n".join(partes) + "\n"


def parametros_tex(codigo) -> str:
    caso = CASOS[codigo]
    claves = [d["clave"] for d in caso.PARAMETROS]
    etiquetas = {d["clave"]: f"{d['etiqueta']}" + (f" [{d['unidad']}]" if d["unidad"] else "") for d in caso.PARAMETROS}
    filas = []
    for k in claves:
        fila = [etiquetas[k]]
        for t in ("NORMAL", "ESTRES", "MAL_CONDICIONADO"):
            fila.append(comun.fmt(caso.ESCENARIOS[t]["parametros"].get(k), 8) if k in caso.ESCENARIOS[t]["parametros"] else "—")
        filas.append(fila)
    return tabla_tex(["Parámetro", "NORMAL", "ESTRÉS", "MAL COND."], filas, "lrrr")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    resultados = {(c, t): simular(c, t) for c, caso in CASOS.items() for t in caso.ESCENARIOS}
    R = lambda c, t: resultados[(c, t)]

    figura("s2_m1.png", [(R("detencion", "ESTRES"), "Velocidad", "v(t)"), (R("detencion", "ESTRES"), "Convergencia de la EDO", "error vs h"), (R("detencion", "MAL_CONDICIONADO"), "Convergencia de las raíces", "raíz múltiple")])
    figura("s2_m2.png", [(R("deterioro", "NORMAL"), "Indicador", "D(t) y umbral"), (R("deterioro", "MAL_CONDICIONADO"), "Convergencia de las raíces", "curva casi tangente"), (R("deterioro", "ESTRES"), "Exceso", "retraso")])
    figura("s2_m3.png", [(R("oscilaciones", "ESTRES"), "Desplazamiento", "x(t), h = 0.5 s"), (R("oscilaciones", "ESTRES"), "Convergencia", "error vs h"), (R("oscilaciones", "MAL_CONDICIONADO"), "Curva de respuesta", "respuesta")])
    figura("s2_m4.png", [(R("sensores", "MAL_CONDICIONADO"), "Señal y reconstrucciones", "Runge"), (R("sensores", "MAL_CONDICIONADO"), "Error de reconstrucción", "error"), (R("sensores", "ESTRES"), "Error de reconstrucción", "ruido σ = 0.002")])
    figura_der()

    salida = {}
    for codigo in CASOS:
        salida[f"errores_{codigo}"] = tabla_errores(codigo, resultados)
        salida[f"conclusiones_{codigo}"] = conclusiones_tex(codigo, resultados)
        salida[f"parametros_{codigo}"] = parametros_tex(codigo)
    salida["m1_ode_estres"] = tabla_spec(R("detencion", "ESTRES"), "Refinamiento del paso")
    salida["m1_ode_mal"] = tabla_spec(R("detencion", "MAL_CONDICIONADO"), "Refinamiento del paso")
    salida["m1_raices_mal"] = tabla_spec(R("detencion", "MAL_CONDICIONADO"), "Raíces de v", ["Método", "t_det [s]", "Iteraciones", "|t − t*| [s]", "|v(t)| residual", "Orden observado"])
    salida["m1_int_mal"] = tabla_spec(R("detencion", "MAL_CONDICIONADO"), "Refinamiento de ∫")
    salida["m1_sens"] = tabla_spec(R("detencion", "NORMAL"), "Sensibilidad")
    salida["m2_newton_normal"] = tabla_spec(R("deterioro", "NORMAL"), "Iteraciones de Newton")
    salida["m2_sens_mal"] = tabla_spec(R("deterioro", "MAL_CONDICIONADO"), "Sensibilidad")
    salida["m3_conv_estres"] = tabla_spec(R("oscilaciones", "ESTRES"), "Refinamiento del paso")
    salida["m3_int_normal"] = tabla_spec(R("oscilaciones", "NORMAL"), "Refinamiento de I", max_filas=7)
    salida["m3_sens_mal"] = tabla_spec(R("oscilaciones", "MAL_CONDICIONADO"), "Sensibilidad")
    salida["m4_ocultos_estres"] = tabla_spec(R("sensores", "ESTRES"), "Reconstrucción de los registros ocultos")
    salida["m4_cond_mal"] = tabla_spec(R("sensores", "MAL_CONDICIONADO"), "Condicionamiento")
    salida["m4_spline_estres"] = tabla_spec(R("sensores", "ESTRES"), "Sistema tridiagonal")
    salida["m4_cruce_normal"] = tabla_spec(R("sensores", "NORMAL"), "Primer cruce", ["Método", "t cruce", "Iteraciones", "|t − t_real|", "|s(t) − u|"])

    # Prueba de persistencia reproducible para el informe.
    ruta = TAB / "prueba_persistencia.db"
    if ruta.exists():
        ruta.unlink()
    filas = []
    for codigo, tipo in [("detencion", "ESTRES"), ("deterioro", "MAL_CONDICIONADO"), ("oscilaciones", "NORMAL"), ("sensores", "ESTRES")]:
        con = persistencia.conectar(ruta)
        sim_id = persistencia.guardar(con, R(codigo, tipo), "Informe S2")
        con.close()
        con = persistencia.conectar(ruta)
        rec = persistencia.recuperar(con, sim_id)
        con.close()
        rep = simular(rec["modulo"], rec["tipo"], rec["parametros"])
        c = persistencia.comparar(rec, rep)
        iters = sum(len(e["iteraciones"]) for e in rec["ejecuciones"])
        filas.append([sim_id, f"M{rec['modulo_id']}", NOMBRE_TIPO[tipo], len(rec["ejecuciones"]), iters, c["valores_comparados"], comun.fmt(c["max_diferencia_relativa"]), "sí" if c["coincide"] else "no"])
    ruta.unlink()
    salida["persistencia"] = tabla_tex(["Sim.", "Módulo", "Escenario", "Métodos", "Filas de iteración", "Valores comparados", "Máx. dif. rel.", "Coincide"], filas, "rllrrrrl")

    for nombre, contenido in salida.items():
        (TAB / f"{nombre}.tex").write_text(contenido, encoding="utf-8")
    print(f"Figuras en {FIG} y {len(salida)} tablas en {TAB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
