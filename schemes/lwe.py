## A Quantum-safe Public-Key-Algorithms Approach with Lattice-based Scheme
## by Bastian Eich, Olaf Grote, Andreas Ahrens

# Learning with Errors - Key generation

import numpy as np
import scipy.stats as stats
from hwcounter import count, count_end


# Saving the current cpu-cycles after importing libraries and before initializing parameters
start = count()

# Defining the relevant parameters
m = 1088    # Number of rows of matrices 'A' and 'e' 
n = 2048    # Number of columns of matrix 'A' and rows of matrix 's'
l = 512     # Number of columns of matrices 's' and 'e'
q = 4096    # Module of 'Z_q'

xs_mu, xs_sigma = (q/2), q      # Shift/compression of gaussian distribution for 'x_s'
xe_mu, xe_sigma = 0, 2          # Shift/compression of gaussian distribution for 'x_e'

# Creating and filling matrix 'A' with pseudo-random numbers from Z_q
A = np.random.choice(q, m*n).reshape(m, n)

# Creating and filling matrix 's' with numbers according to x_s
s = stats.truncnorm.rvs((0 - xs_mu) / xs_sigma, (q - xs_mu) / xs_sigma, loc=xs_mu, scale=xs_sigma, size=n*l).astype(int).reshape(n, l)

# Creating and filling matrix 'e' with numbers according to x_e
e = stats.truncnorm.rvs(-2, 2, loc=xe_mu, scale=xe_sigma, size=m*l).astype(int).reshape(m, l)


# Calculating B = A * s + e mod q
B = np.matmul(A, s)
B = np.add(B, e)
B = np.mod(B, q)

# Calculating the elapsed cpu-cycles
elapsed = count_end()-start

# Writing elapsed cpu-cycles in .txt-file
with open('results_lwe_time.txt', 'a') as f:
  f.write('%d \n' % elapsed)
