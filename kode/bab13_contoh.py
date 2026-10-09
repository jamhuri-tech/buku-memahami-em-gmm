"""Memeriksa setiap bilangan Contoh Soal Bab 13."""
from fractions import Fraction as F

import numpy as np

from bab01_data import awal_mini, data_mini
from bab13_hmm import emisi, maju_mundur

rho = [F(9), F(3), F(3), F(1, 3), F(1, 3), F(1, 9)]
A = [[F(3, 4), F(1, 4)], [F(1, 4), F(3, 4)]]
b = [(r, F(1)) for r in rho]

# Contoh Soal 13.1: peubah maju
al = [[F(1, 2) * b[0][0], F(1, 2) * b[0][1]]]
antara = []
for t in range(1, 6):
    aA = [sum(al[-1][j] * A[j][k] for j in range(2)) for k in range(2)]
    antara.append(aA)
    al.append([b[t][k] * aA[k] for k in range(2)])
assert antara == [[F(7, 2), F(3, 2)], [F(66, 8), F(30, 8)], [F(312, 16), F(144, 16)],
                  [F(57, 8), F(67, 8)], [F(31, 8), F(55, 8)]]
assert al == [[F(9, 2), F(1, 2)], [F(21, 2), F(3, 2)], [F(99, 4), F(15, 4)],
              [F(13, 2), F(9)], [F(19, 8), F(67, 8)], [F(31, 72), F(55, 8)]]
assert F(27, 8) + F(1, 8) == F(7, 2) and F(9, 8) + F(3, 8) == F(3, 2)
P = sum(al[-1])
assert P == F(526, 72) == F(263, 36)

# Contoh Soal 13.2: peubah mundur dan gamma
be = [[F(1), F(1)]]
for t in range(4, -1, -1):
    be.insert(0, [sum(A[k][j] * b[t + 1][j] * be[0][j] for j in range(2))
                  for k in range(2)])
assert be[4] == [F(1, 3), F(7, 9)] and be[3] == [F(5, 18), F(11, 18)]
assert be[2] == [F(2, 9), F(13, 27)]
g3 = al[2][0] * be[2][0] / P
g4 = al[3][0] * be[3][0] / P
assert g3 == F(198, 263) and g4 == F(65, 263) and g3 + g4 == 1
assert round(float(g3), 4) == 0.7529 and round(float(g4), 4) == 0.2471
X, g = data_mini()
pi, mu0, S0 = awal_mini()
gam, xi, _ = maju_mundur(emisi(X, mu0, S0), pi, np.array([[.75, .25], [.25, .75]]))
assert np.allclose(gam[2:4, 0], [float(g3), float(g4)])
assert round(gam[1, 0], 2) == 0.89

# Contoh Soal 13.3: xi_3(1, 2) dan A akhir
x312 = al[2][0] * A[0][1] * b[3][1] * be[3][1] / P
assert x312 == F(1089, 2104) and round(float(x312), 4) == 0.5176
assert np.isclose(xi[2, 0, 1], float(x312))
z = [0, 0, 0, 1, 1, 1]
n = np.zeros((2, 2))
for t in range(5):
    n[z[t], z[t + 1]] += 1
assert np.allclose(n / n.sum(1, keepdims=True), [[2 / 3, 1 / 3], [0, 1]])
assert 2 ** 600 > 10 ** 80
print("Contoh Soal Bab 13: semua bilangan cocok")
