# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt

# Define physical constants and parameters
F = 96485  # Faraday constant in C/mol
R = 8.314  # Universal gas constant in J/(mol*K)
T = 298    # Temperature in Kelvin (25°C)

# Define Butler-Volmer parameters
i0 = 1e-6         # Exchange current density in A (adjustable)
alpha = 0.5       # Charge transfer coefficient (typically between 0 and 1)
n = 2             # Number of electrons transferred in dopamine oxidation

# Create an array of overpotentials (η) in Volts
eta = np.linspace(-0.5, 0.5, 400)

# Calculate the current using the Butler-Volmer equation
# Equation: i = i0 * [exp((alpha * n * F * eta) / (R*T)) - exp(-(1-alpha) * n * F * eta/(R*T))]
current = i0 * (np.exp((alpha * n * F * eta) / (R * T)) - np.exp(-(1 - alpha) * n * F * eta / (R * T)))

# Create a plot of current vs. overpotential
plt.figure(figsize=(8, 5))
plt.plot(eta, current, label='Butler-Volmer Response', color='blue')
plt.xlabel('Overpotential, η (V)')
plt.ylabel('Current, i (A)')
plt.title('Simulated Butler-Volmer Oxidation Experiment')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
