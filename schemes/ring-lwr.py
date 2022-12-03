# Bastian Eich - Masterthesis
# RLWR - Schlüsselerzeugung

import numpy as np
import scipy.stats as stats
from hwcounter import count, count_end
from sys import getsizeof


# Speichern des aktuellen CPU-Zyklus nach Einbindung der Bibliotheken und vor Initialisierung der Parameter
start = count()

m = 2048   # Anzahl der Zeilen von 'A' (= der in 'A' enthaltenen Polynome)
k = 1088   # Anzahl der Spalten von 'A' und 's' (= der in den Polynomen von 'A' und 's' enthaltenen Koeffizienten)
q = 4096  # Modul des Zahlenraums Z_q 
p = 1024   # Modul der Teilmenge Z_p, muss kleiner sein als q

o = [1] + [0] * (k - 1) + [1] # Polynomdivision durch 'o' stellt sicher, dass der höchste Koeffizient n-1 entspricht

xs_mu, xs_sigma = (q/2), q    # Verschiebung/Stauchung der Normalenverteilung für x_s
xe_mu, xe_sigma = 1, 1          # Verschiebung/Stauchung der Normalenverteilung für x_e

# Erzeugung und Befüllung von A mit pseudozufälligen Zahlen aus Z_q
A = np.random.choice(q, m*k).astype(int).reshape(m, k)

# Erzeugung und Befüllung von s mit pseudozufälligen Zahlen aus x_s
s = stats.truncnorm.rvs((0 - xs_mu) / xs_sigma, (q - xs_mu) / xs_sigma, loc=xs_mu, scale=xs_sigma, size=k).astype(int)

# Initialisierung der für die Schleife notwendigen Parameter
i = 0
B = np.zeros(m*k).astype(int).reshape(m, k)

# B = floor((A * s) * (p/q)) mod p für jede Zeile von A
while i < m:
    b = np.polymul(A[i], s)%q
    # Sicherstellen, dass der maximale Grad des Polynoms nicht überschritten wird
    b = np.floor(np.polydiv(b, o)[1]) 
    b = np.mod(np.floor((p/q)*b), p).astype(int)
    # Auffüllen der Matrix 'b' für den Fall, dass nach der Division 0x^3 oder 0x^2 im Polynom auftauchen
    while len(b) < k:
        b = np.insert(b,0,0)
    B[i] = b
    i += 1

# Ermittlung der benötigten CPU-Zyklen nach abschließender Berechnung von B
elapsed = count_end()-start

# Ermittlung der Schlüsselgrößen
size_s = getsizeof(s)
size_A = getsizeof(A)
size_B = getsizeof(B)

# Schreiben der benötigten CPU-Zyklen in ein .txt-File
with open('results_rlwr_time.txt', 'a') as f:
  f.write('%d \n' % elapsed)

# Schreiben der Schlüsselgrößen in ein .txt-File
with open('results_rlwr_size.txt', 'a') as g:
  g.write('s: %d  ' % size_s)
  g.write('A: %d  ' % size_A)
  g.write('B: %d  \n' % size_B)