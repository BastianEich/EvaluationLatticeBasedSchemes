## A Quantum-safe Public-Key-Algorithms Approach with Lattice-based Scheme
## by Bastian Eich, Olaf Grote, Andreas Ahrens

# Ring Learning with Errors 2 - Key generation

import numpy as np
import scipy.stats as stats
from hwcounter import count, count_end


# Saving the current cpu-cycles after importing libraries and before initializing parameters
start = count()

# Defining the relevant parameters
m = 1024    # Number of rows of matrices 'A' and 'e' (= number of polynomials per matrix)
k = 1024    # Number of coefficients in the polynomials of matrices 'A', 's' and 'e' ((k-1) is the highest grade of polynomial)
q = 4096    # Module of 'Z_q'

o = [1] + [0] * (k - 1) + [1]   # Polynomial division by 'o' ensures, that (n-1) stay the highest coefficient

xs_mu, xs_sigma = (q/2), q      # Shift/compression of gaussian distribution for 'x_s'
xe_mu, xe_sigma = 0, 2          # Shift/compression of gaussian distribution for 'x_e'

# Creating and filling matrix 'A' with pseudo-random numbers from Z_q
A = np.random.choice(q, m*k).astype(int).reshape(m, k)

# Creating and filling matrix 's' with numbers according to x_s
s = stats.truncnorm.rvs((0 - xs_mu) / xs_sigma, (q - xs_mu) / xs_sigma, loc=xs_mu, scale=xs_sigma, size=k).astype(int)

# Creating and filling matrix 'e' with numbers according to x_e
e = stats.truncnorm.rvs(-2, 2, loc=xe_mu, scale=xe_sigma, size=m*k).astype(int).reshape(m, k)

# Initializing parameters necessary for while-loop
i = 0
B = np.zeros(m*k).astype(int).reshape(m, k)

# Calculating B = A * s + e mod q for every row of 'A'
while i < m:
    b = np.polymul(A[i], s)%q
    b = np.floor(np.polydiv(b, o)[1]) 
    b = (np.polyadd(b,e[i])%q).astype(int)
    # Fill up matrix 'B' in case the first digits of the outcome are zeroes
    while len(b) < k:
        b = np.insert(b,0,0) 
    B[i] = b
    i += 1

# Calculating the elapsed cpu-cycles
elapsed = count_end()-start

# Writing elapsed cpu-cycles in .txt-file
with open('results_rlwe2_time.txt', 'a') as f:
  f.write('%d \n' % elapsed)
