## A Quantum-safe Public-Key-Algorithms Approach with Lattice-based Scheme
## by Bastian Eich, Olaf Grote, Andreas Ahrens

# Module Learning with Errors - Key generation

import numpy as np
import scipy.stats as stats
from hwcounter import count, count_end


# Saving the current cpu-cycles after importing libraries and before initializing parameters
start = count()

# Defining the relevant parameters
m = 5       # Number of rows of matrices 'A' and 'e' (= polynomials per column n)
k = 256     # Number of coefficients in the polynomials of matrices 'A', 's' and 'e' ((k-1) is the highest grade of polynomial)
n = 5       # Number of columns of matrix 'A' and rows of matrix 's'
q = 16384   # Module of 'Z_q'

o = [1] + [0] * (k - 1) + [1]   # Polynomial division by 'o' ensures, that (n-1) stay the highest coefficient

xs_mu, xs_sigma = (q/2), q      # Shift/compression of gaussian distribution for 'x_s'
xe_mu, xe_sigma = 0, 2          # Shift/compression of gaussian distribution for 'x_e'

# Creating and filling matrix 'A' with pseudo-random numbers from Z_q
A = np.random.choice(q, n*m*k).astype(int).reshape(n, m, k)

# Creating and filling matrix 's' with numbers according to x_s
s = stats.truncnorm.rvs((0 - xs_mu) / xs_sigma, (q - xs_mu) / xs_sigma, loc=xs_mu, scale=xs_sigma, size=n*k).astype(int).reshape(n, k)

# Creating and filling matrix 'e' with numbers according to x_e
e = stats.truncnorm.rvs(-2, 2, loc=xe_mu, scale=xe_sigma, size=m*k).astype(int).reshape(m, k)

# Initializing parameters necessary for while-loop
i = 0
B = np.zeros(m*k).astype(int).reshape(m, k)
x = np.zeros(n*k).astype(int).reshape(n, k)

# Calculating B = A * s + e mod q for every row of every array in matrix 'A'
while i < m:
    j = 0
    while j < n:
        b = np.polymul(A[j][i], s[j])%q 
        b = np.floor(np.polydiv(b, o)[1])
        # Fill up matrix 'B' in case the first digits of the outcome are zeroes
        while len(b) < k:
            b = np.insert(b,0,0)    
        if j == 0:
            x[j] = b
        else:
            x[j] = (np.polyadd(x[j-1], b)%q).astype(int)     
        j += 1
    B[i] = (np.polyadd(x[j-1],e[i])%q).astype(int) 
    i += 1

# Calculating the elapsed cpu-cycles
elapsed = count_end()-start

# Writing elapsed cpu-cycles in .txt-file
with open('results_mlwe_time.txt', 'a') as f:
  f.write('%d \n' % elapsed)
