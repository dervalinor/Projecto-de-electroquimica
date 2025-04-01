import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Parámetros para la ecuación de Butler-Volmer en condiciones ideales ---
alpha = 0.5       # Coeficiente de transferencia de carga
n = 2             # Número de electrones transferidos en la reacción de oxidación de dopamina
F = 96485         # Constante de Faraday (C/mol)
R = 8.314         # Constante de gas universal (J/(mol·K))
T = 298           # Temperatura (K)
k0 = 1e-2         # Constante de velocidad estándar (cm/s)
A = 0.07          # Área del electrodo (cm²)
C_bulk = 1e-6     # Concentración de dopamina en el bulk (mol/cm³)
# Para condiciones ideales se considerarán únicamente los términos de la ecuación de Butler-Volmer sin difusividad ni adsorción
E0 = 0.2         # Potencial estándar de reducción para dopamina (V vs Ag/AgCl)

# --- Parámetros para el barrido de voltaje (FSCV) ---
scan_rate = 400   # Velocidad de barrido (V/s), típica en FSCV
E_start = -0.4    # Potencial inicial (V)
E_vertex = 1.3    # Potencial de inversión (V)
time_total = 0.01 # Tiempo total del experimento (s)

# --- Generación del vector de tiempo ---
dt = 1e-5                    # Paso de tiempo (s)
t = np.arange(0, time_total, dt)
num_points = len(t)          # Número total de puntos en la simulación

# --- Generación del barrido de potencial triangular ---
E = np.zeros(num_points)
half_time = time_total / 2
for i in range(num_points):
    if t[i] < half_time:
        E[i] = E_start + (E_vertex - E_start) * t[i] / half_time
    else:
        E[i] = E_vertex - (E_vertex - E_start) * (t[i] - half_time) / half_time

# --- Modelo ideal de Butler-Volmer (sin difusión ni adsorción) ---
def butler_volmer_ideal(E_applied):
    # Calcular el sobrepotencial: diferencia entre el potencial aplicado y el potencial estándar
    eta = E_applied - E0
    # Términos exponenciales para la oxidación y reducción
    exp_term_a = np.exp(-alpha * n * F * eta / (R * T))
    exp_term_c = np.exp((1 - alpha) * n * F * eta / (R * T))
    # Corriente de intercambio ideal (sin contribución de difusión o adsorción adicional)
    j0 = n * F * k0 * C_bulk
    # Corriente neta por transferencia de carga, según Butler-Volmer
    j_net = j0 * (exp_term_c - exp_term_a)
    return j_net

# --- Cálculo de la corriente total en condiciones ideales ---
j = np.zeros(num_points)
for i in range(num_points):
    j[i] = butler_volmer_ideal(E[i])
current = j * A * 1e6  # Convertir la densidad de corriente (A/cm²) a corriente en microamperios (µA)

# --- Gráfica en condiciones ideales ---
plt.figure(figsize=(10, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# --- Creación de un colormap personalizado para el gradiente (opcional) ---
colors = [(0, 'blue'), (0.5, 'purple'), (1, 'red')]
cmap = LinearSegmentedColormap.from_list('dopamine_cmap', colors, N=100)
color_idx = np.linspace(0, 1, num_points)

# --- Gráfica principal: Voltamograma cíclico ideal ---
plt.subplot(2, 1, 1)
for i in range(num_points-1):
    plt.plot(E[i:i+2], current[i:i+2], color=cmap(color_idx[i]), linewidth=1.5, alpha=0.8)
plt.xlabel('Potencial (V vs. Ag/AgCl)', fontsize=12)
plt.ylabel('Corriente (μA)', fontsize=12)
plt.title('FSCV Ideal: Oxidación de Dopamina', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=E0, color='k', linestyle='--', alpha=0.3)
plt.annotate('E₀', (E0, 0), xytext=(E0+0.05, 0.1*np.max(current)),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), fontsize=10)

# --- Añadir marcadores para los picos de oxidación/reducción ---
idx_ox = np.argmax(current)
idx_red = np.argmin(current)
plt.plot(E[idx_ox], current[idx_ox], 'o', color='red', markersize=8)
plt.plot(E[idx_red], current[idx_red], 'o', color='blue', markersize=8)
plt.annotate('Pico de oxidación', (E[idx_ox], current[idx_ox]),
             xytext=(E[idx_ox]+0.1, current[idx_ox]), fontsize=10)
plt.annotate('Pico de reducción', (E[idx_red], current[idx_red]),
             xytext=(E[idx_red]-0.4, current[idx_red]), fontsize=10)

# --- Gráfica secundaria: Rampa de potencial triangular ---
plt.subplot(2, 1, 2)
plt.plot(t*1000, E, color='darkgreen', linewidth=2)
plt.xlabel('Tiempo (ms)', fontsize=12)
plt.ylabel('Potencial aplicado (V)', fontsize=12)
plt.title('Rampa de Potencial Triangular', fontsize=12)
plt.grid(True, alpha=0.3)

# --- Ajustes finales del diseño ---
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# --- Añadir leyenda informativa con parámetros usados ---
textstr = '\n'.join((
    r'Parámetros:',
    r'$\alpha=%.1f$' % (alpha, ),
    r'$n=%d$ electrones' % (n, ),
    r'$C_{bulk}=%.1f$ \textmu M' % (C_bulk*1e6, ),
    r'Velocidad de barrido: %d V/s' % (scan_rate, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
plt.gcf().text(0.15, 0.02, textstr, fontsize=10, bbox=props)

# --- Guardar la figura en alta resolución ---
plt.savefig('dopamine_fscv_ideal.png', dpi=300, bbox_inches='tight')
plt.savefig('dopamine_fscv_ideal.pdf', format='pdf', bbox_inches='tight')

print("Simulación en condiciones ideales completada y gráfica guardada como 'dopamine_fscv_ideal.png'")
