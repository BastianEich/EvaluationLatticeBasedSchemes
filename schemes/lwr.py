# Bastian Eich - Masterthesis
# LWR - Schlüsselerzeugung

import numpy as np
import scipy.stats as stats
from hwcounter import count, count_end
from sys import getsizeof


# Speichern des aktuellen CPU-Zyklus nach Einbindung der Bibliotheken und vor Initialisierung der Parameter
start = count()

# Definition der Eingangsparameter
m = 1088   # Anzahl der Zeilen von 'A' 
n = 2048   # Anzahl der Spalten von 'A' und Zeilen von 's'
l = 512   # Anzahl der Spalten von 's'
q = 4096  # Modul des Zahlenraums Z_q
p = 1024   # Modul der Teilmenge Z_p, muss kleiner sein als q

xs_mu, xs_sigma = (q/2), q    # Verschiebung/Stauchung der Normalenverteilung für x_s

# Erzeugung und Befüllung von A mit pseudozufälligen Zahlen aus Z_q
A = np.random.choice(q, m*n).reshape(m, n)

# Erzeugung und Befüllung von s mit pseudozufälligen Zahlen aus x_s
s = stats.truncnorm.rvs((0 - xs_mu) / xs_sigma, (q - xs_mu) / xs_sigma, loc=xs_mu, scale=xs_sigma, size=n*l).astype(int).reshape(n, l)

# Berechnung von B = floor((A * s) * (p/q)) mod p 
B = np.matmul(A, s)
B = np.mod(np.floor((p/q)*B), p).astype(int) # Rundungsfunktion, die bei LWR den Parameter "e" ersetzt

# Ermittlung der benötigten CPU-Zyklen nach abschließender Berechnung von B
elapsed = count_end()-start

# Ermittlung der Schlüsselgrößen
size_s = getsizeof(s)
size_A = getsizeof(A)
size_B = getsizeof(B)

# Schreiben der benötigten CPU-Zyklen in ein .txt-File
with open('results_lwr_time.txt', 'a') as f:
  f.write('%d \n' % elapsed)

# Schreiben der Schlüsselgrößen in ein .txt-File
with open('results_lwr_size.txt', 'a') as g:
  g.write('s: %d  ' % size_s)
  g.write('A: %d  ' % size_A)
  g.write('B: %d  \n' % size_B)