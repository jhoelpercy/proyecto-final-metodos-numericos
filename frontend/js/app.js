/**
 * SIMTELE-RISK · aplicación web (semana 3).
 * Rutas: #/ inicio y panel · #/modulo/:id · #/simulacion/:id · #/historial
 * Modo API (FastAPI + SQLite) o modo estático (datos/experimentos.json, sin historial).
 */
(function () {
  "use strict";

  const TIPOS = { NORMAL: "Normal", ESTRES: "Estrés", MAL_CONDICIONADO: "Mal condicionado" };
  const $ = (sel, raiz = document) => raiz.querySelector(sel);
  const app = $("#app");
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const estado = { modo: "api", modulos: [], estatico: null, panel: null, graficos: [], ultimo: null };

  // ---------- utilidades ----------
  function num(v, cifras = 5) {
    if (v === null || v === undefined || v === "") return "—";
    if (typeof v === "string") return v;
    if (typeof v === "boolean") return v ? "Sí" : "No";
    if (Array.isArray(v)) return vector(v);
    if (!Number.isFinite(v)) return "no finito";
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (Number.isInteger(v) && a < 1e7) return String(v);
    return a < 1e-3 || a >= 1e6 ? v.toExponential(3) : String(Number(v.toPrecision(cifras)));
  }
  function vector(v) {
    const partes = v.slice(0, 3).map((x) => num(x, 4));
    return `[${partes.join(", ")}${v.length > 3 ? `, … (${v.length})` : ""}]`;
  }
  async function pedir(url, opciones = {}) {
    const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opciones });
    let cuerpo = null;
    try { cuerpo = await r.json(); } catch (_) { /* sin cuerpo */ }
    if (!r.ok) {
      const d = cuerpo && cuerpo.detail;
      const error = new Error((d && d.mensaje) || `HTTP ${r.status}`);
      error.problemas = (d && d.problemas) || [];
      throw error;
    }
    return cuerpo;
  }
  function aviso(texto) {
    const n = document.createElement("div");
    n.className = "aviso-flotante";
    n.setAttribute("role", "status");
    n.textContent = texto;
    document.body.append(n);
    setTimeout(() => n.remove(), 3200);
  }
  function destruirGraficos() {
    estado.graficos.forEach((g) => g.destroy());
    estado.graficos = [];
  }
  function modulo(id) {
    return estado.modulos.find((m) => String(m.id) === String(id) || m.codigo === id);
  }
  function etiquetaTipo(tipo) {
    return `<span class="etiqueta tipo-${tipo}">${TIPOS[tipo] || esc(tipo)}</span>`;
  }

  // ---------- arranque ----------
  async function iniciar() {
    try {
      const ctl = new AbortController();
      const t = setTimeout(() => ctl.abort(), 2500);
      await pedir("/api/salud", { signal: ctl.signal });
      clearTimeout(t);
      estado.modulos = await pedir("/api/modulos");
      $("#modo").textContent = "Semana 3 · API + SQLite";
    } catch (_) {
      estado.modo = "estatico";
      $("#modo").textContent = "Semana 3 · modo estático";
      try {
        estado.estatico = await pedir("datos/experimentos.json");
        estado.modulos = estado.estatico.modulos;
      } catch (e) {
        app.innerHTML = `<p class="error">No se pudo conectar con la API ni leer datos/experimentos.json (${esc(e.message)}). Ejecute <code>python -m uvicorn backend.simtele.api:app</code> desde la raíz del repositorio.</p>`;
        return;
      }
    }
    try { estado.panel = (await pedir("datos/panel.json")).panel; } catch (_) { estado.panel = null; }
    window.addEventListener("hashchange", enrutar);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => estado.ultimo && estado.ultimo());
    enrutar();
  }

  function enrutar() {
    destruirGraficos();
    const [ruta, consulta] = (location.hash.slice(1) || "/").split("?");
    const params = new URLSearchParams(consulta || "");
    const partes = ruta.split("/").filter(Boolean);
    let activa = "inicio";
    if (partes[0] === "modulo" && modulo(partes[1])) {
      activa = `modulo-${partes[1]}`;
      vistaModulo(modulo(partes[1]), params.get("tipo") || "NORMAL");
    } else if (partes[0] === "simulacion" && partes[1]) {
      activa = "historial";
      vistaSimulacion(Number(partes[1]), params.get("relanzar") === "1");
    } else if (partes[0] === "historial") {
      activa = "historial";
      vistaHistorial(params);
    } else {
      vistaInicio();
    }
    document.querySelectorAll("#navegacion a").forEach((a) => a.classList.toggle("activa", a.dataset.ruta === activa));
    app.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }

  // ---------- inicio y panel general ----------
  function vistaInicio() {
    estado.ultimo = null;
    const tarjetas = estado.modulos.map((m) => `
      <a class="tarjeta tarjeta-caso" href="#/modulo/${m.id}">
        <span class="numero">Módulo ${m.id}</span>
        <h2>${esc(m.nombre)}</h2>
        <p>${esc(m.pregunta)}</p>
        <ul class="metodos">${m.metodos.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      </a>`).join("");
    app.innerHTML = `
      <h1>Cuatro casos de estudio, un mismo análisis de errores</h1>
      <p class="bajada">Cada módulo resuelve su modelo con los métodos pertinentes en tres condiciones —NORMAL, ESTRÉS y MAL CONDICIONADO—, mide errores contra una referencia analítica o de alta precisión y separa la conclusión numérica de la conclusión sobre el caso. ${estado.modo === "api" ? "Las ejecuciones se guardan en SQLite y pueden recuperarse y relanzarse desde el historial." : "<strong>Modo estático:</strong> se muestran los 12 escenarios precalculados; para cambiar parámetros y guardar el historial ejecute la API."}</p>
      <section class="tarjetas-casos" aria-label="Módulos">${tarjetas}</section>
      <section class="tarjeta seccion" aria-labelledby="t-panel">
        <div class="estado-matriz"><h2 id="t-panel">Panel general de comparación (parámetros por defecto)</h2><span class="muted pequeno">Generado con <code>python -m backend.simtele.experimentos</code></span></div>
        ${panelHtml()}
      </section>
      <section class="tarjeta seccion" aria-labelledby="t-cobertura">
        <h2 id="t-cobertura">Cobertura de métodos por módulo</h2>
        ${coberturaHtml()}
      </section>`;
  }

  function panelHtml() {
    if (!estado.panel) return `<p class="muted">No se encontró datos/panel.json.</p>`;
    const filas = estado.modulos.map((m) => {
      const celdas = Object.keys(TIPOS).map((t) => {
        const p = estado.panel[m.codigo] && estado.panel[m.codigo][t];
        if (!p) return `<td class="matriz-celda">—</td>`;
        const fallos = p.convergencias.filter((c) => !c[2]).map((c) => c[1]);
        return `<td class="matriz-celda"><ul>${p.panel.map(([k, v]) => `<li><b>${esc(k)}:</b> ${esc(v)}</li>`).join("")}</ul>
          ${fallos.length ? `<span class="etiqueta estado-no">No converge: ${esc(fallos.join(", "))}</span>` : `<span class="etiqueta estado-si">Todos convergen</span>`}
          <div><a href="#/modulo/${m.id}?tipo=${t}">Abrir escenario →</a></div></td>`;
      }).join("");
      return `<tr><td class="fila-caso">M${m.id} · ${esc(m.nombre)}</td>${celdas}</tr>`;
    }).join("");
    return `<div class="tabla-scroll" tabindex="0" role="region" aria-label="Panel de comparación"><table>
      <thead><tr><th>Módulo</th>${Object.entries(TIPOS).map(([k, v]) => `<th><span class="etiqueta tipo-${k}">${v}</span></th>`).join("")}</tr></thead>
      <tbody>${filas}</tbody></table></div>`;
  }

  function coberturaHtml() {
    const familias = {
      "Teoría de errores": [1, 2, 3, 4], "Sistemas lineales": [2, 4], "Interpolación": [4], "Raíces": [1, 2, 4], "Integración": [1, 3], "EDO (Heun, RK4)": [1, 3],
    };
    return `<div class="tabla-scroll" tabindex="0" role="region" aria-label="Cobertura"><table>
      <thead><tr><th>Familia</th>${estado.modulos.map((m) => `<th>M${m.id} ${esc(m.nombre)}</th>`).join("")}</tr></thead>
      <tbody>${Object.entries(familias).map(([f, ms]) => `<tr><td>${f}</td>${estado.modulos.map((m) => `<td>${ms.includes(m.id) ? "✓" : ""}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div><p class="nota">Sistemas lineales: Jacobi, Gauss-Seidel, LU y gradiente conjugado (M2 calibración; M4 sistema tridiagonal del spline). RK2 declarado: método de Heun.</p>`;
  }

  // ---------- módulo ----------
  function vistaModulo(m, tipo) {
    if (!TIPOS[tipo]) tipo = "NORMAL";
    const esc_ = m.escenarios[tipo];
    app.innerHTML = `
      <p class="muted pequeno"><a href="#/">Inicio</a> / Módulo ${m.id}</p>
      <h1>${esc(m.nombre)}</h1>
      <p class="bajada">${esc(m.pregunta)}</p>
      <section class="encabezado-modulo seccion">
        <div class="tarjeta">
          <h2>Modelo</h2>
          <ul class="formula-lista">${m.modelo.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>
          <div class="columnas">
            <div><h3>Variables y unidades</h3><ul>${m.variables.map((v) => `<li>${esc(v)}</li>`).join("")}</ul></div>
            <div><h3>Hipótesis</h3><p>${esc(m.hipotesis)}</p><h3>Referencia</h3><p>${esc(m.referencia)}</p></div>
          </div>
          <h3>Métodos aplicados</h3><ul class="metodos">${m.metodos.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
        </div>
        <div class="tarjeta">
          <h2>Escenario</h2>
          <div class="selector" role="group" aria-label="Escenario">${Object.entries(TIPOS).map(([k, v]) => `<button type="button" data-tipo="${k}" aria-pressed="${k === tipo}">${v}</button>`).join("")}</div>
          <p>${esc(esc_.descripcion)}</p>
          ${tipo === "MAL_CONDICIONADO" ? `<div class="banda banda-alerta">Subexperimento numérico: muestra sensibilidad matemática. No equivale a riesgo físico ni a falta de convergencia.</div>` : ""}
          <p class="muted pequeno"><b>Fuente de los datos:</b> ${esc(esc_.fuente)}</p>
        </div>
      </section>
      <section class="tarjeta seccion" aria-labelledby="t-param">
        <h2 id="t-param">Parámetros (${TIPOS[tipo]})</h2>
        <form id="formulario" novalidate>
          <div class="rejilla-parametros">${m.parametros.filter((p) => !p.escenarios || p.escenarios.includes(tipo)).map((p) => campo(p, esc_.parametros[p.clave])).join("")}</div>
          <div id="errores-form"></div>
          <div class="acciones">
            <button class="boton" type="submit" id="ejecutar">Ejecutar simulación</button>
            <button class="boton secundario" type="button" id="restablecer">Valores del escenario</button>
            ${estado.modo === "api" ? `<label class="pequeno"><input type="checkbox" id="guardar" checked> Guardar en historial</label><input id="usuario" type="text" maxlength="80" placeholder="Usuario o equipo (opcional)" aria-label="Usuario o equipo">` : `<span class="muted pequeno">Modo estático: sólo valores por defecto.</span>`}
          </div>
        </form>
      </section>
      <div id="resultado"></div>`;

    $(".selector").addEventListener("click", (e) => {
      const b = e.target.closest("[data-tipo]");
      if (b) location.hash = `#/modulo/${m.id}?tipo=${b.dataset.tipo}`;
    });
    $("#restablecer").addEventListener("click", () => vistaModulo(m, tipo));
    $("#formulario").addEventListener("submit", (e) => { e.preventDefault(); ejecutar(m, tipo); });
    ejecutar(m, tipo, true);
  }

  function campo(p, valor) {
    const id = `p-${p.clave}`;
    const unidad = p.unidad ? ` <span>[${esc(p.unidad)}]</span>` : "";
    if (p.opciones) {
      return `<div class="parametro"><label for="${id}">${esc(p.etiqueta)}${unidad}</label>
        <select id="${id}" name="${p.clave}" class="selector-campo">${p.opciones.map((o) => `<option value="${esc(o)}" ${o === valor ? "selected" : ""}>${esc(o)}</option>`).join("")}</select></div>`;
    }
    const rango = `${p.min !== undefined ? `mín ${num(p.min)}` : ""}${p.max !== undefined ? ` · máx ${num(p.max)}` : ""}${p.entero ? " · entero" : ""}`;
    return `<div class="parametro"><label for="${id}">${esc(p.etiqueta)}${unidad}</label>
      <input id="${id}" name="${p.clave}" type="number" inputmode="decimal" required value="${esc(valor)}" ${p.min !== undefined ? `min="${p.min}"` : ""} ${p.max !== undefined ? `max="${p.max}"` : ""} step="${p.entero ? 1 : "any"}">
      <small>${rango}</small></div>`;
  }

  function leerFormulario(m, tipo) {
    const valores = {};
    const problemas = [];
    m.parametros.filter((p) => !p.escenarios || p.escenarios.includes(tipo)).forEach((p) => {
      const nodo = $(`#p-${p.clave}`);
      if (!nodo) return;
      if (p.opciones) { valores[p.clave] = nodo.value; return; }
      const texto = nodo.value.trim();
      const v = Number(texto);
      if (texto === "" || !Number.isFinite(v)) problemas.push(`${p.etiqueta}: valor numérico requerido.`);
      else if (p.min !== undefined && v < p.min) problemas.push(`${p.etiqueta}: mínimo ${num(p.min)}.`);
      else if (p.max !== undefined && v > p.max) problemas.push(`${p.etiqueta}: máximo ${num(p.max)}.`);
      else if (p.entero && !Number.isInteger(v)) problemas.push(`${p.etiqueta}: debe ser entero.`);
      valores[p.clave] = v;
    });
    return { valores, problemas };
  }

  function mostrarProblemas(problemas) {
    $("#errores-form").innerHTML = problemas.length ? `<div class="banda banda-mal" role="alert"><b>Revise los parámetros:</b><ul class="lista-simple">${problemas.map((p) => `<li>${esc(p)}</li>`).join("")}</ul></div>` : "";
  }

  async function ejecutar(m, tipo, inicial = false) {
    const { valores, problemas } = leerFormulario(m, tipo);
    mostrarProblemas(problemas);
    if (problemas.length) return;
    const boton = $("#ejecutar");
    boton.disabled = true;
    boton.textContent = "Calculando…";
    destruirGraficos();
    $("#resultado").innerHTML = `<p class="cargando">Ejecutando métodos…</p>`;
    try {
      let r;
      if (estado.modo === "api") {
        const guardar = !inicial && $("#guardar") && $("#guardar").checked;
        r = await pedir("/api/simular", { method: "POST", body: JSON.stringify({ modulo: m.codigo, tipo, parametros: valores, guardar, usuario_equipo: $("#usuario") ? $("#usuario").value : "" }) });
      } else {
        const defecto = m.escenarios[tipo].parametros;
        const distinto = Object.keys(valores).some((k) => String(valores[k]) !== String(defecto[k]) && Number(valores[k]) !== Number(defecto[k]));
        if (distinto) {
          mostrarProblemas(["En modo estático sólo están disponibles los valores por defecto. Ejecute la API para parámetros personalizados."]);
          r = estado.estatico.resultados[m.codigo][tipo];
        } else r = estado.estatico.resultados[m.codigo][tipo];
      }
      const pintar = () => { destruirGraficos(); $("#resultado").innerHTML = resultadoHtml(r); conectarResultado(r); };
      estado.ultimo = pintar;
      pintar();
      if (r.simulacion_id) aviso(`Simulación #${r.simulacion_id} guardada en el historial.`);
    } catch (e) {
      $("#resultado").innerHTML = "";
      mostrarProblemas(e.problemas && e.problemas.length ? e.problemas : [e.message]);
    } finally {
      boton.disabled = false;
      boton.textContent = "Ejecutar simulación";
    }
  }

  // ---------- resultados ----------
  function resultadoHtml(r, cabecera = "") {
    const ind = r.indicadores.map((i) => `<div class="indicador"><div class="e">${esc(i.etiqueta)}</div><div class="v">${esc(num(i.valor, 6))}${i.unidad ? ` <small>${esc(i.unidad)}</small>` : ""}</div>${i.nota ? `<div class="n">${esc(i.nota)}</div>` : ""}</div>`).join("");
    const ejec = r.ejecuciones.map((e, i) => {
      const met = e.resumen.metricas;
      const err = met.error_abs ?? met.error_abs_d_fren ?? met.error_max_x ?? met.error_max_oculto ?? met.error_abs_muestras_rk4 ?? null;
      return `<tr><td>${esc(e.familia)}</td><td>${esc(e.etiqueta)}</td><td><span class="etiqueta ${e.convergio ? "estado-si" : "estado-no"}">${e.convergio ? "✓ Sí" : "✕ No"}</span></td>
        <td class="num">${e.total_iteraciones}</td><td class="num">${num(e.tolerancia)}</td><td class="num">${num(e.paso)}</td><td class="num">${num(err)}</td><td class="num">${num(e.tiempo_ms, 3)}</td>
        <td class="envuelto">${esc(e.criterio)}</td><td><button class="boton secundario" type="button" data-iter="${i}">Iteraciones</button></td></tr>`;
    }).join("");
    const tablas = r.tablas.map((t) => `<section class="tarjeta seccion"><h2>${esc(t.titulo)}</h2>
      <div class="tabla-scroll alta" tabindex="0" role="region" aria-label="${esc(t.titulo)}"><table><thead><tr>${t.columnas.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${t.filas.map((f) => `<tr>${f.map((v) => `<td class="${typeof v === "number" || /^[-−]?[\d.]+(e[-+]?\d+)?$/.test(String(v)) ? "num" : ""}">${esc(num(v, 8))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      ${t.nota ? `<p class="nota">${esc(t.nota)}</p>` : ""}</section>`).join("");
    const graficos = r.graficos.map((g, i) => `<figure class="tarjeta" style="margin:0"><h3>${esc(g.titulo)}</h3><div class="lienzo"><canvas id="g-${i}" role="img" aria-label="${esc(g.titulo)}"></canvas></div>${g.nota ? `<figcaption class="nota-grafico">${esc(g.nota)}</figcaption>` : ""}</figure>`).join("");
    return `${cabecera}
      <section class="tarjeta seccion">
        <div class="estado-matriz"><h2>Resultados · ${etiquetaTipo(r.tipo)} ${r.es_subexperimento ? `<span class="etiqueta">subexperimento numérico</span>` : ""}</h2>
          <span class="muted pequeno">${r.simulacion_id ? `Simulación <a href="#/simulacion/${r.simulacion_id}">#${r.simulacion_id}</a> · ` : ""}modelo v${esc(r.modelo_version)} · ${num(r.tiempo_total_ms, 4)} ms de cómputo</span></div>
        <p>${esc(r.descripcion)}</p>
        <div class="indicadores">${ind}</div>
      </section>
      <section class="seccion rejilla-graficos" aria-label="Gráficos">${graficos}</section>
      <section class="tarjeta seccion" aria-labelledby="t-ejec">
        <h2 id="t-ejec">Métodos ejecutados</h2>
        <div class="tabla-scroll" tabindex="0" role="region" aria-label="Métodos ejecutados"><table>
          <thead><tr><th>Familia</th><th>Método</th><th>¿Convergió?</th><th>Iteraciones / pasos</th><th>Tolerancia</th><th>Paso h</th><th>Error principal</th><th>t [ms]</th><th>Criterio de parada</th><th></th></tr></thead>
          <tbody>${ejec}</tbody></table></div>
        <div id="detalle-iter"></div>
      </section>
      ${tablas}
      <section class="seccion conclusiones" aria-label="Interpretación">
        <div class="tarjeta conclusion numerica"><h2>Conclusión numérica</h2><p>${esc(r.conclusiones.numerica)}</p></div>
        <div class="tarjeta conclusion fisica"><h2>Conclusión del problema</h2><p>${esc(r.conclusiones.problema)}</p><p class="muted pequeno" style="margin-top:.6rem"><b>Límites:</b> ${esc(r.conclusiones.limites)}</p></div>
      </section>`;
  }

  function conectarResultado(r) {
    r.graficos.forEach((g, i) => dibujar($(`#g-${i}`), g));
    app.querySelectorAll("[data-iter]").forEach((b) => b.addEventListener("click", () => mostrarIteraciones(r.ejecuciones[Number(b.dataset.iter)])));
  }

  function mostrarIteraciones(e) {
    const claves = [];
    e.iteraciones.slice(0, 20).forEach((it) => Object.keys(it.aproximacion).forEach((k) => { if (!claves.includes(k)) claves.push(k); }));
    const filas = e.iteraciones.map((it) => `<tr><td class="num">${it.numero}</td>${claves.map((k) => `<td class="num">${esc(num(it.aproximacion[k], 10))}</td>`).join("")}
      <td class="num">${num(it.residual)}</td><td class="num">${num(it.error_abs)}</td><td class="num">${num(it.error_rel)}</td></tr>`).join("");
    const met = Object.entries(e.resumen.metricas).map(([k, v]) => `<li><b>${esc(k)}:</b> ${esc(num(v, 8))}</li>`).join("");
    const val = Object.entries(e.resumen.valor).map(([k, v]) => `<li><b>${esc(k)}:</b> ${esc(num(v, 12))}</li>`).join("");
    $("#detalle-iter").innerHTML = `<div class="seccion">
      <div class="barra-tabla"><h3 style="margin:0">Tabla de iteraciones · ${esc(e.etiqueta)}</h3><span class="muted pequeno">${e.iteraciones.length < e.total_iteraciones ? `Se muestran ${e.iteraciones.length} de ${e.total_iteraciones} filas (submuestreo uniforme, primera y última incluidas).` : `${e.total_iteraciones} filas.`}</span></div>
      <div class="columnas"><div><h3>Valor</h3><ul>${val}</ul></div><div><h3>Métricas de error</h3><ul>${met}</ul></div><div><h3>Referencia</h3><p>${esc(e.resumen.referencia)}</p><h3>Interpretación</h3><p>${esc(e.resumen.interpretacion)}</p></div></div>
      <div class="tabla-scroll alta" tabindex="0" role="region" aria-label="Iteraciones"><table><thead><tr><th>#</th>${claves.map((k) => `<th>${esc(k)}</th>`).join("")}<th>Residual</th><th>Error abs.</th><th>Error rel.</th></tr></thead><tbody>${filas}</tbody></table></div></div>`;
    $("#detalle-iter").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function dibujar(lienzo, g) {
    const paleta = [css("--referencia"), css("--serie-1"), css("--serie-2"), css("--serie-3"), css("--malcond"), css("--serie-4")];
    const texto = css("--texto-2");
    const rejilla = css("--rejilla");
    const conjuntos = g.series.map((s, i) => {
      const color = paleta[s.color ?? (i % 5) + 1] || paleta[1];
      const puntos = s.estilo !== "linea";
      return {
        label: s.nombre,
        data: s.puntos.map(([x, y]) => ({ x, y })),
        showLine: s.estilo !== "puntos",
        borderColor: color,
        backgroundColor: color,
        borderWidth: s.estilo === "puntos" ? 0 : 2,
        borderDash: s.punteada ? [6, 4] : [],
        pointRadius: puntos ? (s.puntos.length > 80 ? 2 : 3.5) : 0,
        pointHoverRadius: 4,
        spanGaps: false,
        order: s.estilo === "puntos" ? 0 : 1,
      };
    });
    const eje = (tipoLog, titulo) => ({
      type: tipoLog ? "logarithmic" : "linear",
      title: { display: true, text: titulo, color: texto },
      grid: { color: rejilla },
      ticks: { color: texto, maxTicksLimit: 8, callback: tipoLog ? (v) => { const l = Math.log10(v); return Number.isInteger(Math.round(l * 1e6) / 1e6) ? `1e${Math.round(l)}` : ""; } : undefined },
    });
    estado.graficos.push(new Chart(lienzo, {
      type: "scatter",
      data: { datasets: conjuntos },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false, parsing: false, normalized: true,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { align: "end", labels: { color: texto, usePointStyle: true, boxWidth: 8, boxHeight: 8 } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: (${num(c.parsed.x, 6)}, ${num(c.parsed.y, 6)})` } },
        },
        scales: { x: eje(g.log_x, g.eje_x), y: eje(g.log_y, g.eje_y) },
      },
    }));
  }

  // ---------- simulación recuperada ----------
  async function vistaSimulacion(id, relanzar) {
    estado.ultimo = null;
    if (estado.modo !== "api") { app.innerHTML = `<p class="error">El historial requiere la API con base de datos.</p>`; return; }
    app.innerHTML = `<p class="cargando">Recuperando la simulación #${id} de la base de datos…</p>`;
    let r;
    try { r = await pedir(`/api/simulaciones/${id}`); } catch (e) { app.innerHTML = `<p class="error">${esc(e.message)}</p>`; return; }
    const m = modulo(r.modulo_id);
    const params = Object.entries(r.parametros).map(([k, v]) => `<li><b>${esc(k)}</b> = ${esc(num(v, 12))}</li>`).join("");
    const cabecera = `
      <p class="muted pequeno"><a href="#/historial">Historial</a> / Simulación #${r.simulacion_id}</p>
      <h1>Simulación #${r.simulacion_id}: ${esc(m.nombre)}</h1>
      <div class="banda banda-info">Recuperada de la base de datos · ${esc(r.fecha.replace("T", " "))} · ${esc(r.usuario_equipo || "sin usuario")} · estado ${esc(r.estado)} · ${r.ejecuciones.length} métodos con sus iteraciones.</div>
      <section class="tarjeta seccion">
        <div class="estado-matriz"><h2>Parámetros guardados ${etiquetaTipo(r.tipo)}</h2>
          <div class="acciones"><button class="boton" type="button" id="relanzar">Relanzar y comparar</button><a class="boton secundario" href="#/modulo/${m.id}?tipo=${r.tipo}">Abrir el módulo</a></div></div>
        <ul class="lista-simple columnas">${params}</ul>
        <p class="muted pequeno">Fuente: ${esc(r.fuente_datos)}</p>
        <div id="comparacion"></div>
      </section>`;
    const pintar = () => { destruirGraficos(); app.innerHTML = resultadoHtml(r, cabecera); conectarResultado(r); $("#relanzar").addEventListener("click", () => relanzarSim(id)); };
    estado.ultimo = pintar;
    pintar();
    if (relanzar) relanzarSim(id);
  }

  async function relanzarSim(id) {
    const destino = $("#comparacion");
    const boton = $("#relanzar");
    boton.disabled = true;
    destino.innerHTML = `<p class="cargando">Repitiendo el cálculo con los parámetros guardados…</p>`;
    try {
      const { comparacion: c } = await pedir(`/api/simulaciones/${id}/relanzar`, { method: "POST", body: JSON.stringify({ guardar: false }) });
      destino.innerHTML = `<div class="banda ${c.coincide ? "banda-bien" : "banda-mal"}" role="status">${c.coincide ? "✓ El recálculo coincide con lo guardado" : "✕ El recálculo NO coincide"}: ${c.valores_comparados} valores comparados (valores, métricas e iteraciones), diferencia relativa máxima ${num(c.max_diferencia_relativa)}; tolerancia |a − b| ≤ ${num(c.atol)} + ${num(c.rtol)}·max(|a|, |b|).</div>
        <div class="tabla-scroll" tabindex="0" role="region" aria-label="Comparación"><table><thead><tr><th>Familia</th><th>Método</th><th>Valores comparados</th><th>Fuera de tolerancia</th><th>Máx. dif. relativa</th><th>Coincide</th></tr></thead>
        <tbody>${c.detalles.map((d) => `<tr><td>${esc(d.familia)}</td><td>${esc(d.metodo)}</td><td class="num">${d.valores_comparados}</td><td class="num">${d.fuera_de_tolerancia}</td><td class="num">${num(d.max_diferencia_relativa)}</td><td>${d.coincide ? "✓" : "✕"}</td></tr>`).join("")}</tbody></table></div>`;
    } catch (e) {
      destino.innerHTML = `<p class="error">${esc(e.message)}</p>`;
    } finally {
      boton.disabled = false;
    }
  }

  // ---------- historial ----------
  async function vistaHistorial(params) {
    estado.ultimo = null;
    if (estado.modo !== "api") {
      app.innerHTML = `<h1>Historial</h1><p class="banda banda-alerta">El historial se guarda en SQLite y requiere la API: <code>python -m uvicorn backend.simtele.api:app --port 8000</code>.</p>`;
      return;
    }
    const q = params.get("q") || "", mod = params.get("modulo") || "", tipo = params.get("tipo") || "", pagina = Math.max(1, Number(params.get("pagina") || 1));
    const limite = 15;
    app.innerHTML = `
      <h1>Historial de simulaciones</h1>
      <p class="bajada">Cada fila es una ejecución completa guardada en la base de datos: parámetros, métodos, iteraciones, resúmenes, tablas y gráficos. Abra una para visualizarla o relánzela para comprobar que el recálculo coincide.</p>
      <section class="tarjeta seccion">
        <form class="filtros" id="filtros">
          <input name="q" type="search" placeholder="Buscar por id, usuario, fecha o parámetro…" value="${esc(q)}" aria-label="Buscar">
          <select name="modulo" aria-label="Módulo"><option value="">Todos los módulos</option>${estado.modulos.map((m) => `<option value="${m.id}" ${String(m.id) === mod ? "selected" : ""}>M${m.id} · ${esc(m.nombre)}</option>`).join("")}</select>
          <select name="tipo" aria-label="Escenario"><option value="">Todos los escenarios</option>${Object.entries(TIPOS).map(([k, v]) => `<option value="${k}" ${k === tipo ? "selected" : ""}>${v}</option>`).join("")}</select>
          <button class="boton" type="submit">Buscar</button>
        </form>
        <div id="lista"><p class="cargando">Consultando…</p></div>
      </section>`;
    $("#filtros").addEventListener("submit", (e) => {
      e.preventDefault();
      const f = new FormData(e.target);
      const u = new URLSearchParams();
      for (const [k, v] of f.entries()) if (v) u.set(k, v);
      location.hash = `#/historial?${u}`;
    });
    try {
      const u = new URLSearchParams({ q, limite, desplazamiento: (pagina - 1) * limite });
      if (mod) u.set("modulo", mod);
      if (tipo) u.set("tipo", tipo);
      const datos = await pedir(`/api/simulaciones?${u}`);
      if (!datos.simulaciones.length) { $("#lista").innerHTML = `<p class="muted">No hay simulaciones guardadas con esos filtros. Ejecute una desde un módulo con «Guardar en historial» marcado.</p>`; return; }
      const paginas = Math.ceil(datos.total / limite);
      const enlace = (p) => { const x = new URLSearchParams(params); x.set("pagina", p); return `#/historial?${x}`; };
      $("#lista").innerHTML = `<div class="tabla-scroll" tabindex="0" role="region" aria-label="Simulaciones"><table>
        <thead><tr><th>#</th><th>Fecha</th><th>Módulo</th><th>Escenario</th><th>Usuario / equipo</th><th>Métodos</th><th>Parámetros</th><th></th></tr></thead>
        <tbody>${datos.simulaciones.map((s) => `<tr><td class="num">${s.id}</td><td>${esc(s.fecha.replace("T", " "))}</td><td>M${s.modulo_id} · ${esc(s.nombre_modulo)}</td><td>${etiquetaTipo(s.tipo)}</td>
          <td>${esc(s.usuario_equipo || "—")}</td><td class="num">${s.ejecuciones}</td><td class="envuelto pequeno">${esc(Object.entries(s.parametros).map(([k, v]) => `${k}=${num(v, 6)}`).join(", "))}</td>
          <td><div class="acciones"><a class="boton secundario" href="#/simulacion/${s.id}">Abrir</a><a class="boton" href="#/simulacion/${s.id}?relanzar=1">Relanzar</a></div></td></tr>`).join("")}</tbody></table></div>
        <div class="paginador">${pagina > 1 ? `<a href="${enlace(pagina - 1)}">← Anterior</a>` : ""}<span>Página ${pagina} de ${paginas} · ${datos.total} simulaciones</span>${pagina < paginas ? `<a href="${enlace(pagina + 1)}">Siguiente →</a>` : ""}</div>`;
    } catch (e) {
      $("#lista").innerHTML = `<p class="error">${esc(e.message)}</p>`;
    }
  }

  iniciar();
})();
