import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
F = 96485          # Faraday constant (C/mol)
R = 8.314          # Gas constant (J/mol*K)
T = 298.15         # Temperature (K)
n = 1              # Number of electrons transferred
alpha = 0.5        # Transfer coefficient
j0 = 1e-6          # Exchange current density (A/cm^2)
A = 0.01           # Electrode area (cm^2)
I0 = j0 * A        # Exchange current (A)
E_eq = 0.0         # Equilibrium potential (V)

# Simulation parameters
t_total = 2.0      # Total time of the experiment (s)
num_points = 1000  # Number of time points
time = np.linspace(0, t_total, num_points)

# Define a cyclic potential waveform:
# A triangular wave: ramp up from -0.5 V to +0.5 V in half the time,
# then ramp down back to -0.5 V in the remaining half.
E_min = -0.5
E_max = 0.5
half_time = t_total / 2

E = np.zeros_like(time)
for i, t in enumerate(time):
    if t <= half_time:
        E[i] = E_min + (E_max - E_min) * (t / half_time)
    else:
        E[i] = E_max - (E_max - E_min) * ((t - half_time) / half_time)

# Compute overpotential at each time step
eta = E - E_eq

# Compute current using the Butler-Volmer equation
# j = j0 [exp((alpha*n*F*eta)/(R*T)) - exp(-((1-alpha)*n*F*eta)/(R*T))]
current_BV = I0 * (np.exp((alpha * n * F * eta) / (R * T)) - np.exp(-(1 - alpha) * n * F * eta / (R * T)))

# Add a diffusion-controlled component (approximated by a Cottrell-like term)
# I_diff(t) = k * (1/sqrt(t)) for t > 0, with k a scaling constant.
# We avoid division by zero by starting from a small time offset.
k = 1e-7
current_diff = np.zeros_like(time)
time_offset = np.copy(time)
time_offset[time_offset == 0] = 1e-6  # avoid division by zero
current_diff = k / np.sqrt(time_offset)

# Total current is the sum of the kinetic and diffusion-controlled contributions.
# (This is a simplified approach for illustration.)
current_total = current_BV + current_diff

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(time, current_total*1e6, label='Total Current', color='blue')
plt.plot(time, current_BV*1e6, '--', label='Kinetic (Butler–Volmer) Contribution', color='red')
plt.plot(time, current_diff*1e6, ':', label='Diffusion (Cottrell-like) Contribution', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Current (µA)')
plt.title('FSCV Simulation: Current Response vs. Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output.png")  # Guarda la imagen como un archivo PNG

