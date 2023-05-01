## A Quantum-safe Public-Key-Algorithms Approach with Lattice-based Scheme
## by Bastian Eich, Olaf Grote, Andreas Ahrens

# Ring Learning with Rounding - Key generation

import numpy as np
import scipy.stats as stats
from hwcounter import count, count_end


# Saving the current cpu-cycles after importing libraries and before initializing parameters
start = count()

# Defining the relevant parameters
m = 2048    # Number of rows of matrix 'A' (= number of polynomials per matrix)
k = 1088    # Number of coefficients in the polynomials of matrices 'A' and 's' ((k-1) is the highest grade of polynomial)
q = 4096    # Module of 'Z_q'
p = 1024    # Module of 'Z_p', needs to be smaller than q

o = [1] + [0] * (k - 1) + [1]   # Polynomial division by 'o' ensures, that (n-1) stay the highest coefficient

xs_mu, xs_sigma = (q/2), q      # Shift/compression of gaussian distribution for 'x_s'

# Creating and filling matrix 'A' with pseudo-random numbers from Z_q
A = np.random.choice(q, m*k).astype(int).reshape(m, k)

# Creating and filling matrix 's' with numbers according to x_s
s = stats.truncnorm.rvs((0 - xs_mu) / xs_sigma, (q - xs_mu) / xs_sigma, loc=xs_mu, scale=xs_sigma, size=k).astype(int)

# Initializing parameters necessary for while-loop
i = 0
B = np.zeros(m*k).astype(int).reshape(m, k)

# Calculating B = floor((A * s) * (p/q)) mod p for every row of 'A'
while i < m:
    b = np.polymul(A[i], s)%q
    b = np.floor(np.polydiv(b, o)[1]) 
    b = np.mod(np.floor((p/q)*b), p).astype(int)
    # Fill up matrix 'B' in case the first digits of the outcome are zeroes
    while len(b) < k:
        b = np.insert(b,0,0)
    B[i] = b
    i += 1

# Calculating the elapsed cpu-cycles
elapsed = count_end()-start

# Writing elapsed cpu-cycles in .txt-file
with open('results_rlwr_time.txt', 'a') as f:
  f.write('%d \n' % elapsed)