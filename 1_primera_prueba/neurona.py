# para poder ejecutar este codigo es necesario crear una entorno:
# python3 -m venv env
# source env/bin/activate
# pip install neuron
# source env/bin/activate
# python3.
# Tambien instalar este paquete: pip3 install matplotlib

from neuron import h, gui         # Importa NEURON y su interfaz gráfica (opcional)
import matplotlib.pyplot as plt   # Importa matplotlib para graficar

# 1. Crear la neurona (soma) y configurar sus propiedades
soma = h.Section(name='soma')     # Se crea un compartimento llamado 'soma'
soma.L = 20                       # Longitud del soma en micrómetros
soma.diam = 20                    # Diámetro del soma en micrómetros
soma.insert('hh')                 # Se inserta el mecanismo de Hodgkin-Huxley (canales iónicos)

# 2. Aplicar un estímulo de corriente usando IClamp
stim = h.IClamp(soma(0.5))        # Coloca el estimulador en el centro del soma (0.5 representa la mitad)
stim.delay = 100                  # El estímulo comienza a los 100 ms
stim.dur = 500                    # Duración del estímulo: 500 ms
stim.amp = 0.1                    # Amplitud del estímulo en nA

# 3. Registrar variables de la simulación: tiempo y voltaje
t_vec = h.Vector().record(h._ref_t)         # Vector para registrar el tiempo de simulación
v_vec = h.Vector().record(soma(0.5)._ref_v)   # Vector para registrar el voltaje en el centro del soma

# 4. Inicializar y ejecutar la simulación
h.finitialize(-65)                # Se inicializa el potencial de membrana a -65 mV (valor de reposo)
h.continuerun(700)                # Se corre la simulación durante 700 ms

# 5. Graficar los resultados con matplotlib
plt.figure(figsize=(8, 4))        # Define el tamaño de la figura
plt.plot(t_vec, v_vec, label='Voltaje en el soma')  # Grafica el voltaje vs tiempo y añade una etiqueta
plt.xlabel("Tiempo (ms)")         # Etiqueta del eje x
plt.ylabel("Voltaje (mV)")        # Etiqueta del eje y
plt.title("Respuesta de la neurona al estímulo de corriente")  # Título de la gráfica
plt.legend()                      # Muestra la leyenda

# 6. Guardar la gráfica en un archivo de imagen (ejemplo: grafica_neurona.png)
plt.savefig("grafica_neurona.png", dpi=300)  # Guarda la imagen con una resolución de 300 dpi

# 7. Mostrar la gráfica en pantalla
plt.show()

