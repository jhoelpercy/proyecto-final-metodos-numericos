"""Definición de los cuatro casos de estudio y de sus tres escenarios.

Planteamiento (semana 1): variables, unidades, hipótesis, referencias y escenarios.
La implementación de cada caso está en backend/simtele/casos/ (semana 3).
"""

from __future__ import annotations

NOTA_SINTETICOS = "Datos sintéticos y parámetros supuestos; escenarios hipotéticos sin relación con fallas reales."

MODULOS = [
    {
        "id": 1,
        "codigo": "detencion",
        "nombre": "Detención de emergencia",
        "pregunta": "¿Cómo varían tiempo y distancia de detención frente a cambios de velocidad, tiempo de reacción y frenado?",
        "modelo": ["dx/dt = v", "dv/dt = −a(v, t)", "x(0) = 0, v(0) = v0", "v(t_det) = 0"],
        "variables": ["x: posición [m]", "v: velocidad [m/s]", "a: desaceleración [m/s²]", "t_r: reacción [s]", "m = 1200 kg"],
        "hipotesis": "Masa puntual, recorrido rectilíneo, reacción a velocidad constante.",
        "metodos": ["Heun", "RK4", "Bisección", "Newton-Raphson", "Secante", "Trapecio", "Simpson 1/3", "Romberg"],
        "referencia": "v(t) = v0 − a t; t_det = v0/a; d = v0²/(2a)",
        "escenarios": {
            "NORMAL": "v0 = 5 m/s, a = 2.5 m/s², t_r = 0.5 s, h = 0.1 s.",
            "ESTRES": "v0 = 7 m/s, a = 1.2 + 0.05 v² m/s², t_r = 1.5 s, h = 0.5 s, tol = 1e-13.",
            "MAL_CONDICIONADO": "Frenado plano v = v0 (1 − t/T)^5: raíz múltiple con derivada nula.",
        },
        "estado": "Planteado",
    },
    {
        "id": 2,
        "codigo": "deterioro",
        "nombre": "Deterioro y mantenimiento",
        "pregunta": "¿Cuándo alcanzaría un indicador hipotético de deterioro un umbral de inspección?",
        "modelo": ["D(t) = D0 + α t + β t²", "D(t) − D_u = 0", "A x = b, A SPD, b = A x*"],
        "variables": ["D: desgaste hipotético [%]", "t: tiempo [meses]", "α [%/mes], β [%/mes²]", "x: coeficientes calibrados"],
        "hipotesis": "Indicador hipotético cuadrático; calibración por tramos con datos sin ruido.",
        "metodos": ["Jacobi", "Gauss-Seidel", "LU", "Gradiente conjugado", "Bisección", "Newton-Raphson", "Secante"],
        "referencia": "Raíz de la cuadrática; solución x* prefijada del sistema SPD.",
        "escenarios": {
            "NORMAL": "A = I + 0.1 L, 6 tramos, κ cercano a 1, tol = 1e-10.",
            "ESTRES": "A = I + 5 L, 40 tramos, tol = 1e-12.",
            "MAL_CONDICIONADO": "Matriz de Hilbert de orden 8, κ >> 1000; curva casi tangente al umbral.",
        },
        "estado": "Sistemas lineales ejecutados",
    },
    {
        "id": 3,
        "codigo": "oscilaciones",
        "nombre": "Oscilaciones por viento",
        "pregunta": "¿Cómo se modifica la oscilación ante cambios de fuerza externa, amortiguamiento y parámetros del modelo?",
        "modelo": ["m x″ + c x′ + k x = F(t)", "x′ = v", "v′ = (F(t) − c v − k x)/m", "I = ∫ x(t)² dt"],
        "variables": ["x: desplazamiento [m]", "v: velocidad [m/s]", "m [kg], k [N/m], c [N·s/m]", "F: fuerza [N]"],
        "hipotesis": "Oscilador lineal de un grado de libertad; viento armónico idealizado.",
        "metodos": ["Heun", "RK4", "Trapecio", "Simpson 1/3", "Romberg"],
        "referencia": "F = 0, subamortiguado: x(t) = e^(−ζω₀t)[C₁cos(ω_d t) + C₂sin(ω_d t)]",
        "escenarios": {
            "NORMAL": "Vibración libre con m = 1200 kg, k = 4800 N/m, ζ = 0.05, h = 0.05 s.",
            "ESTRES": "Ráfaga armónica F0 = 1500 N cerca de resonancia, paso extremo h = 0.5 s.",
            "MAL_CONDICIONADO": "ζ = 0.005 y frecuencia en el flanco de la resonancia: alta sensibilidad.",
        },
        "estado": "Planteado",
    },
    {
        "id": 4,
        "codigo": "sensores",
        "nombre": "Reconstrucción de sensores",
        "pregunta": "¿Cómo reconstruir registros faltantes y medir la distorsión que producen datos con ruido?",
        "modelo": ["y(t) conocida en N instantes", "Ocultar puntos de prueba", "p(t_i) = y_i", "e = |p(t_oculto) − y(t_oculto)|"],
        "variables": ["t: tiempo [s]", "y: señal normalizada [u]", "σ: ruido [u]", "N: muestras"],
        "hipotesis": "Señal sintética conocida; ruido gaussiano con semilla fija.",
        "metodos": ["Lagrange", "Newton (diferencias divididas)", "Spline cúbico natural", "Gradiente conjugado, Gauss-Seidel, Jacobi y LU (sistema del spline)", "Bisección", "Newton-Raphson", "Secante"],
        "referencia": "Señal analítica evaluada en los puntos ocultos.",
        "escenarios": {
            "NORMAL": "y = sin(t) con 17 muestras, sin ruido.",
            "ESTRES": "Senoidal amortiguada con 31 muestras y ruido σ = 0.002.",
            "MAL_CONDICIONADO": "Pulso de Runge con 21 nodos uniformes: distribución desfavorable.",
        },
        "estado": "Planteado",
    },
]
