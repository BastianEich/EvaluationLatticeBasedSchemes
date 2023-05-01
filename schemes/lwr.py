## A Quantum-safe Public-Key-Algorithms Approach with Lattice-based Scheme
## by Bastian Eich, Olaf Grote, Andreas Ahrens

# Learning with Rounding - Key generation

import numpy as np
import scipy.stats as stats
from hwcounter import count, count_end


# Saving the current cpu-cycles after importing libraries and before initializing parameters
start = count()

# Defining the relevant parameters
m = 1088    # Number of rows of matrix 'A'
n = 2048    # Number of columns of matrix 'A' and rows of matrix 's'
l = 512     # Number of columns of matrix 's'
q = 4096    # Module of 'Z_q'
p = 1024    # Module of 'Z_p', needs to be smaller than q

xs_mu, xs_sigma = (q/2), q    # Shift/compression of gaussian distribution for 'x_s'

# Creating and filling matrix 'A' with pseudo-random numbers from Z_q
A = np.random.choice(q, m*n).reshape(m, n)

# Creating and filling matrix 's' with numbers according to x_s
s = stats.truncnorm.rvs((0 - xs_mu) / xs_sigma, (q - xs_mu) / xs_sigma, loc=xs_mu, scale=xs_sigma, size=n*l).astype(int).reshape(n, l)

#Calculating B = floor((A * s) * (p/q)) mod p 
B = np.matmul(A, s)
B = np.mod(np.floor((p/q)*B), p).astype(int)

# Calculating the elapsed cpu-cycles
elapsed = count_end()-start

# Writing elapsed cpu-cycles in .txt-file
with open('results_lwr_time.txt', 'a') as f:
  f.write('%d \n' % elapsed)
