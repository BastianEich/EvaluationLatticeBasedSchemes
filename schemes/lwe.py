# Bastian Eich - Masterthesis
# LWE - Schlüsselerzeugung

import numpy as np
import scipy.stats as stats
from hwcounter import count, count_end
from sys import getsizeof


# Speichern des aktuellen CPU-Zyklus nach Einbindung der Bibliotheken und vor Initialisierung der Parameter
start = count()

# Definition der Eingangsparameter
m = 1088   # Anzahl der Zeilen von 'A' und Zeilen von 'e' 
n = 2048   # Anzahl der Spalten von 'A' und Zeilen von 's'
l = 512   # Anzahl der Spalten von 's' und 'e'
q = 4096  # Modul des Zahlenraums Z_q

xs_mu, xs_sigma = (q/2), q    # Verschiebung/Stauchung der Normalenverteilung für x_s
xe_mu, xe_sigma = 0, 2          # Verschiebung/Stauchung der Normalenverteilung für x_e

# Erzeugung und Befüllung von A mit pseudozufälligen Zahlen aus Z_q
A = np.random.choice(q, m*n).reshape(m, n)

# Erzeugung und Befüllung von s mit pseudozufälligen Zahlen aus x_s
s = stats.truncnorm.rvs((0 - xs_mu) / xs_sigma, (q - xs_mu) / xs_sigma, loc=xs_mu, scale=xs_sigma, size=n*l).astype(int).reshape(n, l)

# Erzeugung und Befüllung von s mit pseudozufälligen Zahlen aus x_e
e = stats.truncnorm.rvs(-2, 2, loc=xe_mu, scale=xe_sigma, size=m*l).astype(int).reshape(m, l)


# Berechnung von B = A * s + e mod q
B = np.matmul(A, s)
B = np.add(B, e)
B = np.mod(B, q)

# Ermittlung der benötigten CPU-Zyklen nach abschließender Berechnung von B
elapsed = count_end()-start

# Ermittlung der Schlüsselgrößen
size_s = getsizeof(s)
size_A = getsizeof(A)
size_B = getsizeof(B)

# Schreiben der benötigten CPU-Zyklen in ein .txt-File
with open('results_lwe_time.txt', 'a') as f:
  f.write('%d \n' % elapsed)

# Schreiben der Schlüsselgrößen in ein .txt-File
with open('results_lwe_size.txt', 'a') as g:
  g.write('s: %d  ' % size_s)
  g.write('A: %d  ' % size_A)
  g.write('B: %d  \n' % size_B)
