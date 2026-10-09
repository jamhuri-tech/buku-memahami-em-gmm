"""Memeriksa setiap bilangan Contoh Soal Bab 7."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import SIGMA0_2, data_mini, langkah_m, tanggung_jawab

X, g = data_mini()
Xi = [tuple(int(v) for v in x) for x in X]

# Contoh Soal 7.1: laju model pi saja
rho = [Fr(9), Fr(3), Fr(3), Fr(1, 3), Fr(1, 3), Fr(1, 9)]
suku = [4 * r / (r + 1) ** 2 for r in rho]
assert suku == [Fr(36, 100), Fr(3, 4), Fr(3, 4), Fr(3, 4), Fr(3, 4), Fr(36, 100)]
Mp = sum(suku) / 6
assert sum(suku) == Fr(372, 100) and Mp == Fr(62, 100)
Ic = 4 * 6
Iobs = sum(4 * (r - 1) ** 2 / (r + 1) ** 2 for r in rho)
assert Ic == 24 and Iobs == Fr(912, 100) and 1 - Iobs / Ic == Mp
assert (Fr(16, 10)) ** 2 == Fr(256, 100)
for r in rho:   # identitas (rho+1)^2 - (rho-1)^2 = 4 rho
    assert (r + 1) ** 2 - (r - 1) ** 2 == 4 * r

# Contoh Soal 7.2: pemulaan simetris
r1 = [1 / (1 + Fr(3) ** ((x[0] - 4) // 3)) for x in Xi]
assert r1 == [Fr(3, 4), Fr(3, 4), Fr(1, 2), Fr(1, 2), Fr(1, 4), Fr(1, 4)]
mu0 = np.array([[3.0, 4.0], [5.0, 4.0]])
d2 = ((X[:, None, :] - mu0[None]) ** 2).sum(-1)
assert np.allclose(d2[:, 0] - d2[:, 1], 4 * (X[:, 0] - 4))
N1 = sum(r1)
m11 = sum(r * x[0] for r, x in zip(r1, Xi)) / N1
m12 = sum(r * x[1] for r, x in zip(r1, Xi)) / N1
assert (m11, m12) == (3, Fr(7, 2))
s11 = sum(r * x[0] ** 2 for r, x in zip(r1, Xi))
s12 = sum(r * x[0] * x[1] for r, x in zip(r1, Xi))
s22 = sum(r * x[1] ** 2 for r, x in zip(r1, Xi))
assert (s11, s12, s22) == (42, 39, 54)
S = [[s11 / 3 - m11 ** 2, s12 / 3 - m11 * m12], [0, s22 / 3 - m12 ** 2]]
assert S[0] == [5, Fr(5, 2)] and S[1][1] == Fr(23, 4)
R, _ = tanggung_jawab(X, np.array([.5, .5]), mu0,
                      np.array([SIGMA0_2 * np.eye(2)] * 2))
pi1, mu1, S1 = langkah_m(X, R)
assert np.allclose(mu1, [[3, 3.5], [5, 4.5]])
assert np.allclose(S1[0], [[5, 2.5], [2.5, 5.75]]) and np.allclose(S1[1], S1[0])
det = Fr(5) * Fr(23, 4) - Fr(25, 4)
assert det == Fr(45, 2)
adj = [[Fr(23, 4), Fr(-5, 2)], [Fr(-5, 2), Fr(5)]]
d = (-2, -1)
w = [sum(adj[a][b] * d[b] for b in range(2)) / det for a in range(2)]
assert [sum(adj[a][b] * d[b] for b in range(2)) for a in range(2)] == [-9, 0]
assert w == [Fr(-2, 5), 0]
assert round(1 / (1 + np.exp(-1.2)), 4) == 0.7685
R2, _ = tanggung_jawab(X, pi1, mu1, S1)
assert np.allclose(R2[2:4, 0], 0.5) and np.isclose(R2[0, 0], 1 / (1 + np.exp(-1.2)))
# simetri (u, v) -> (u, u - v)
D = [(a - 4, b - 4) for a, b in Xi]
T = [(u, u - v) for u, v in D]
assert [D.index(t) for t in T] == [1, 0, 3, 2, 5, 4]

# Contoh Soal 7.3: partisi {1, 2, 4}
P = [Xi[0], Xi[1], Xi[3]]
m = (Fr(sum(p[0] for p in P), 3), Fr(sum(p[1] for p in P), 3))
assert m == (2, 4)
dv = [(p[0] - m[0], p[1] - m[1]) for p in P]
assert dv == [(-1, -3), (-1, 0), (2, 3)]
Sx = [[sum(a[i] * a[j] for a in dv) / 3 for j in range(2)] for i in range(2)]
assert Sx == [[2, 3], [3, 6]] and Sx[0][0] * Sx[1][1] - Sx[0][1] ** 2 == 3

# teks: digit per iterasi, peluang pemulaan, sisa kenaikan
assert round(-np.log10(0.5), 1) == 0.3 and round(-np.log10(0.99), 3) == 0.004
assert round(1 - 0.745 ** 10, 2) == 0.95 and round(1 - 0.425 ** 5, 3) == 0.986
assert round(0.62 / 0.38, 1) == 1.6 and round(0.99 / 0.01) == 99
print("Contoh Soal Bab 7: semua bilangan cocok")
