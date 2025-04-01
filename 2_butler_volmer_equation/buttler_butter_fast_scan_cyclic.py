import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import odeint

# --- Parámetros para la ecuación de Butler-Volmer ---
alpha = 0.5  # Coeficiente de transferencia de carga
n = 2  # Número de electrones transferidos en la reacción electroquímica
F = 96485  # Constante de Faraday (C/mol), carga de un mol de electrones
R = 8.314  # Constante de gas universal (J/(mol·K))
T = 298  # Temperatura en Kelvin (aproximadamente 25°C)
k0 = 1e-2  # Constante de velocidad estándar de reacción (cm/s)
A = 0.07  # Área del electrodo (cm²), afecta la magnitud de la corriente medida
C_bulk = 1e-6  # Concentración de dopamina en la solución (mol/cm³)
D = 6e-6  # Coeficiente de difusión de dopamina (cm²/s)
delta = 10e-4  # Espesor de la capa de difusión (cm)
E0 = 0.2  # Potencial estándar de reducción de dopamina (V vs Ag/AgCl)

# --- Parámetros del barrido de voltaje (FSCV) ---
scan_rate = 400  # Velocidad de barrido en V/s (típica para FSCV)
E_start = -0.4  # Potencial inicial (V)
E_vertex = 1.3  # Potencial de inversión (V)
time_total = 0.01  # Tiempo total del experimento (s)

# --- Generación del vector de tiempo ---
dt = 1e-5  # Paso de tiempo (s)
t = np.arange(0, time_total, dt)
num_points = len(t)  # Número total de puntos en la simulación

# --- Generación del barrido de potencial triangular ---
E = np.zeros(num_points)
half_time = time_total / 2
for i in range(num_points):
    if t[i] < half_time:
        E[i] = E_start + (E_vertex - E_start) * t[i] / half_time
    else:
        E[i] = E_vertex - (E_vertex - E_start) * (t[i] - half_time) / half_time

# --- Modelo de Butler-Volmer para la densidad de corriente ---
def butler_volmer(E_applied, t_idx):
    eta = E_applied - E0  # Sobrepotencial
    exp_term_a = np.exp(-alpha * n * F * eta / (R * T))  # Término anódico
    exp_term_c = np.exp((1 - alpha) * n * F * eta / (R * T))  # Término catódico

    j0 = n * F * k0 * C_bulk  # Corriente de intercambio
    j_net = j0 * (exp_term_c - exp_term_a)  # Corriente neta por transferencia de carga

    j_diffusion = (n * F * D * C_bulk / delta) * (1 - np.exp(-scan_rate * t[t_idx] / (D / delta**2)))  # Modelo de difusión Randles-Sevcik
    j_adsorption = 0.15 * j0 * np.sin(np.pi * E_applied / E_vertex) * (E_applied > 0)  # Adsorción superficial característica de dopamina

    return float(j_net + j_diffusion + j_adsorption)

# --- Cálculo de la corriente total ---
j = np.zeros(num_points)
for i in range(num_points):
    j[i] = butler_volmer(E[i], i)
current = j * A * 1e6  # Convertir a microamperios (µA)

# --- Añadir ruido realista a la señal ---
noise_level = 0.05 * np.max(np.abs(current))  # Nivel de ruido basado en la señal
noise = np.random.normal(0, noise_level, num_points)  # Ruido gaussiano
current_noisy = current + noise  # Señal con ruido agregado

# --- Configuración de la figura y el estilo de la gráfica ---
plt.figure(figsize=(10, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# --- Creación de un colormap para el gradiente de color en la gráfica ---
colors = [(0, 'blue'), (0.5, 'purple'), (1, 'red')]
cmap = LinearSegmentedColormap.from_list('dopamine_cmap', colors, N=100)
color_idx = np.linspace(0, 1, num_points)

# --- Gráfica del voltamograma cíclico ---
plt.subplot(2, 1, 1)
for i in range(num_points-1):
    plt.plot(E[i:i+2], current_noisy[i:i+2], color=cmap(color_idx[i]), linewidth=1.5, alpha=0.8)
plt.xlabel('Potencial (V vs. Ag/AgCl)', fontsize=12)
plt.ylabel('Corriente (μA)', fontsize=12)
plt.title('Fast-Scan Cyclic Voltammetry: Oxidación de Dopamina', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=E0, color='k', linestyle='--', alpha=0.3)
plt.annotate('E₀', (E0, 0), xytext=(E0+0.05, 0.1*np.max(current_noisy)),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), fontsize=10)

# --- Añadir marcadores de picos de oxidación y reducción ---
idx_ox = np.argmax(current_noisy)
idx_red = np.argmin(current_noisy)
plt.plot(E[idx_ox], current_noisy[idx_ox], 'o', color='red', markersize=8)
plt.plot(E[idx_red], current_noisy[idx_red], 'o', color='blue', markersize=8)
plt.annotate('Pico de oxidación', (E[idx_ox], current_noisy[idx_ox]),
             xytext=(E[idx_ox]+0.1, current_noisy[idx_ox]), fontsize=10)
plt.annotate('Pico de reducción', (E[idx_red], current_noisy[idx_red]),
             xytext=(E[idx_red]-0.4, current_noisy[idx_red]), fontsize=10)

# --- Gráfica del voltaje vs tiempo ---
plt.subplot(2, 1, 2)
plt.plot(t*1000, E, color='darkgreen', linewidth=2)
plt.xlabel('Tiempo (ms)', fontsize=12)
plt.ylabel('Potencial aplicado (V)', fontsize=12)
plt.title('Rampa de potencial triangular', fontsize=12)
plt.grid(True, alpha=0.3)

# --- Ajustes finales ---
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# --- Guardado de la figura ---
plt.savefig('dopamine_fscv_simulation.png', dpi=300, bbox_inches='tight')
plt.savefig('dopamine_fscv_simulation.pdf', format='pdf', bbox_inches='tight')

print("Simulación completada y gráfica guardada como 'dopamine_fscv_simulation.png'")
