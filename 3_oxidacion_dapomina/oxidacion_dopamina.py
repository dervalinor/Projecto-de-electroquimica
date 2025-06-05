#Para instalar los paquetes necesarios debemos crear un entorno para
#poder instalar los paquetes de python necesarios

#python3 -m venv venv
#source venv/bin/activate

# Simulación de la oxidación de dopamina bajo estrés oxidativo
import numpy as np
import matplotlib.pyplot as plt

# Parámetros de simulación
T = 3600       # tiempo total en segundos (1 hora)
dt = 0.1       # paso de integración en segundos
n_pas = int(T/dt)

# Variables de estado iniciales: concentraciones
dopamina = np.zeros(n_pas+1)        # [Dopamina] (DA)
peroxido_hidrogeno = np.zeros(n_pas+1)  # [H2O2]
quinona = np.zeros(n_pas+1)         # [Dopamina-quinona]

dopamina[0] = 1.0           # conc. inicial de dopamina (u.a.)
peroxido_hidrogeno[0] = 0.1  # conc. inicial de H2O2 (u.a.)

# Fluctuaciones estocásticas del pH (rango fisiológico ~6.5–7.8)
ph = np.zeros(n_pas+1)
ph[0] = 7.2                   # pH inicial medio
theta = 0.05                  # velocidad de reversión hacia el medio (OU)
sigma_ph = 0.1                # magnitud de la fluctuación
for i in range(n_pas):
    ph[i+1] = ph[i] + theta*(7.2 - ph[i])*dt + sigma_ph*np.sqrt(dt)*np.random.randn()
ph = np.clip(ph, 6.5, 7.8)    # limitar al rango fisiológico

# Fluctuaciones estocásticas de la concentración de oxígeno disuelto
oxigeno = np.zeros(n_pas+1)
oxigeno[0] = 0.20             # fracción inicial de O2 ambiental (20%)
theta_o2 = 0.05
sigma_o2 = 0.05
for i in range(n_pas):
    oxigeno[i+1] = oxigeno[i] + theta_o2*(0.20 - oxigeno[i])*dt + sigma_o2*np.sqrt(dt)*np.random.randn()
oxigeno = np.clip(oxigeno, 0.1, 0.3)  # rango plausible [10%, 30%]

# Constantes cinéticas
k_base = 0.001    # constante de velocidad base de oxidación
k_decomp = 0.001  # velocidad de degradación enzimática de H2O2

# Función dependiente de pH (aproximamos un aumento ~10^ΔpH)
def factor_ph(pH):
    return 10**(pH - 7.2)

# Definir picos de estrés oxidativo externos (añadir H2O2 repentinamente)
n_picos = 5
tiempos_picos = np.sort(np.random.uniform(0, T, n_picos))
magnitudes_picos = np.random.uniform(0.1, 0.5, n_picos)
indices_picos = [int(tp/dt) for tp in tiempos_picos]

# Integración del sistema de EDOs (método de Euler explícito)
for i in range(n_pas):
    # Velocidad instantánea de oxidación DA -> quinona + H2O2
    v = k_base * dopamina[i] * oxigeno[i] * factor_ph(ph[i])
    # Ecuaciones diferenciales
    d_dop = -v
    d_h2o2 = v - k_decomp * peroxido_hidrogeno[i]
    d_quin = v
    # Actualizar concentraciones
    dopamina[i+1] = dopamina[i] + d_dop * dt
    peroxido_hidrogeno[i+1] = peroxido_hidrogeno[i] + d_h2o2 * dt
    quinona[i+1] = quinona[i] + d_quin * dt
    # Agregar picos aleatorios de H2O2
    if i in indices_picos:
        j = indices_picos.index(i)
        peroxido_hidrogeno[i+1] += magnitudes_picos[j]

# Graficar resultados de [DA] y [H2O2] vs tiempo
plt.figure(figsize=(8,5))
plt.plot(np.arange(n_pas+1)*dt, dopamina, label='[Dopamina]')
plt.plot(np.arange(n_pas+1)*dt, peroxido_hidrogeno, label='[H$_2$O$_2$]')
plt.xlabel('Tiempo (s)')
plt.ylabel('Concentración (u.a.)')
plt.legend()
plt.title('Dinámica de [DA] y [H$_2$O$_2$] con variabilidad estocástica')
plt.tight_layout()
plt.show()

