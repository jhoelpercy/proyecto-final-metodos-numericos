# SIMTELE-RISK · Semana 2

Desafío integrador de **Métodos Numéricos** (UMSA, FCPN, La Paz). Aplicación web para estudiar cuatro casos de un sistema de transporte por cable, cada uno en tres escenarios. Proyecto académico con **datos sintéticos**, inspirado en Mi Teleférico de La Paz: no describe fallas verificadas ni se conecta con equipos reales.

- Informe de la semana 2: [`docs/informe_semana2.pdf`](docs/informe_semana2.pdf)
- Informe de la semana 1: [`docs/informe_semana1.pdf`](docs/informe_semana1.pdf)

## Integrantes

| Nombre | CI | RU | Mención |
|---|---|---|---|
| Saúl Walter Callisaya Sumi | 15074779 | 1885982 | Ingeniería de Sistemas |
| Ignacio Josue Zarate Camargo | 12392989 | 1886051 | IA y Ciencia de Datos |
| Luis Alejandro Zeballos Quiroz | 12896709 | 1886052 | IA y Ciencia de Datos |
| Jhoel Percy González Herrera | 9990880 | 1886116 | Ingeniería de Sistemas |

## Casos de estudio y métodos

| Módulo | Pregunta | Métodos |
|---|---|---|
| M1 · Detención de emergencia | ¿Cuánto tarda y recorre la cabina hasta detenerse? | Heun (RK2), RK4 con evento v = 0; bisección, Newton, secante; trapecio, Simpson 1/3, Romberg |
| M2 · Deterioro y mantenimiento | ¿Cuándo el deterioro alcanza el umbral? | Bisección, Newton, secante; Jacobi, Gauss-Seidel, LU, gradiente conjugado |
| M3 · Oscilaciones por viento | ¿Cómo oscila la cabina ante ráfagas? | Heun (RK2), RK4; trapecio, Simpson 1/3, Romberg |
| M4 · Reconstrucción de sensores | ¿Cómo reconstruir registros perdidos? | Lagrange, Newton (diferencias divididas), spline cúbico natural; CG/GS/Jacobi/LU para el spline; bisección, Newton, secante |

Escenarios en **todos** los módulos: **NORMAL**, **ESTRÉS** y **MAL CONDICIONADO** (subexperimento numérico). La teoría de errores (absoluto, relativo, porcentual, residual, sensibilidad, orden observado) es transversal. RK2 declarado: **método de Heun**.

## Ejecución

Requiere Python 3.10 o superior.

```bash
pip install -r backend/requirements.txt
python -m backend.simtele.experimentos          # corre los 12 escenarios y genera frontend/datos/
python -m uvicorn backend.simtele.api:app --port 8000
```

Abrir <http://localhost:8000/>. Las simulaciones se guardan en `database/simtele.db` (configurable con la variable `SIMTELE_DB`) y sobreviven al reinicio del servidor.

**Modo estático** (sin API ni historial, con los 12 escenarios precalculados): `python -m http.server 8000` desde la raíz y abrir <http://localhost:8000/frontend/>. Sirve también para GitHub Pages.

### API

| Ruta | Función |
|---|---|
| `GET /api/modulos` | Modelos, parámetros con rangos y escenarios |
| `POST /api/simular` | `{"modulo", "tipo", "parametros", "guardar", "usuario_equipo"}` → resultado completo (422 si los parámetros no son válidos) |
| `GET /api/simulaciones?q=&modulo=&tipo=` | Historial con búsqueda |
| `GET /api/simulaciones/{id}` | Recupera una simulación desde la base |
| `POST /api/simulaciones/{id}/relanzar` | Repite el cálculo y compara con lo guardado dentro de la tolerancia |

## Pruebas

```bash
python -m pytest        # 54 pruebas
```

Incluyen pruebas con solución analítica de cada familia de métodos, los 12 escenarios, el esquema SQL y la prueba obligatoria de persistencia: ejecutar → guardar → cerrar → recuperar → repetir → coincidir.

## Estructura

```
backend/simtele/
  numerico/        errores, sistemas (S1); raices, integracion, edo, interpolacion (S2)
  casos/           detencion, deterioro, oscilaciones, sensores + comun (validación, tablas, gráficos)
  persistencia.py  SQLite: guardar, listar, recuperar, comparar
  api.py           FastAPI + servidor de la aplicación
  experimentos.py  12 escenarios → frontend/datos/experimentos.json y panel.json
backend/tests/     pruebas de métodos, casos, persistencia y API
database/          schema.sql (6 entidades, v2) y su prueba
frontend/          index.html, css/, js/app.js, datos/, vendor/chart.umd.js
docs/              informes, figuras y generar_informe_s2.py (tablas y figuras del informe)
render.yaml        despliegue como servicio web de Python
```

Para regenerar el informe: `pip install -r docs/requirements.txt`, `python docs/generar_informe_s2.py` y compilar `docs/informe_semana2.tex` con XeLaTeX (dos pasadas).

## Notas

- Los valores cercanos a la precisión de máquina (≈ 10⁻¹⁵) pueden variar levemente según el equipo.
- Las tablas de iteraciones guardan como máximo 400 filas por método (submuestreo uniforme con la primera y la última); el total de iteraciones se conserva.
