"""Memeriksa setiap bilangan Contoh Soal Bab 4."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import awal_mini, data_mini, langkah_m, tanggung_jawab

X, g = data_mini()
Xi = [tuple(int(v) for v in x) for x in X]
r1 = [Fr(9, 10), Fr(3, 4), Fr(3, 4), Fr(1, 4), Fr(1, 4), Fr(1, 10)]
r2 = [1 - r for r in r1]

# Contoh Soal 4.1: langkah M
N1, N2 = sum(r1), sum(r2)
assert N1 == N2 == 3
s2 = sum(r * x[0] for r, x in zip(r2, Xi))
assert s2 == Fr(159, 10) and s2 / 3 == Fr(53, 10)
assert sum(r * x[1] for r, x in zip(r2, Xi)) == Fr(159, 10)
assert [x[0] ** 2 for x in Xi] == [1, 1, 16, 16, 49, 49]
assert [x[0] * x[1] for x in Xi] == [1, 4, 4, 28, 28, 49]
assert [x[1] ** 2 for x in Xi] == [1, 16, 1, 49, 16, 49]
m11 = sum(r * x[0] ** 2 for r, x in zip(r1, Xi))
m12 = sum(r * x[0] * x[1] for r, x in zip(r1, Xi))
m22 = sum(r * x[1] ** 2 for r, x in zip(r1, Xi))
assert m11 == m22 == Fr(348, 10) and m12 == Fr(258, 10)
mu = Fr(27, 10)
S11, S12 = m11 / 3 - mu * mu, m12 / 3 - mu * mu
assert mu * mu == Fr(729, 100)
assert S11 == Fr(431, 100) and S12 == Fr(131, 100)
R, _ = tanggung_jawab(X, *awal_mini())
pi1, mu1, Sig1 = langkah_m(X, R)
assert np.allclose(Sig1[0], [[4.31, 1.31], [1.31, 4.31]])
assert np.allclose(Sig1[1], Sig1[0]) and np.allclose(mu1[1], [5.3, 5.3])
d = Fr(13, 10)
assert S11 + d * d == 6 and S12 + d * d == 3

# Contoh Soal 4.2: langkah E iterasi kedua
lam1, lam2 = S11 + S12, S11 - S12
assert lam1 == Fr(562, 100) and lam2 == 3
u = lambda v: (Fr((v[0] + v[1]) ** 2, 2) if True else 0, Fr((v[0] - v[1]) ** 2, 2))
def d2(x, m):
    v = (x[0] - m, x[1] - m)
    a, b = u(v)
    return a / lam1 + b / lam2
m1, m2 = Fr(27, 10), Fr(53, 10)
assert u((Fr(-17, 10), Fr(-17, 10)))[0] == Fr(578, 100)
assert round(float(d2(Xi[0], m1)), 5) == 1.02847
assert u((Fr(-43, 10), Fr(-43, 10)))[0] == Fr(3698, 100)
assert round(float(d2(Xi[0], m2)), 5) == 6.58007
a = float(d2(Xi[0], m2) - d2(Xi[0], m1)) / 2
assert round(a, 5) == 2.77580 and round(1 / (1 + np.exp(-a)), 5) == 0.94135
assert u((Fr(13, 10), Fr(-17, 10))) == (Fr(8, 100), Fr(45, 10))
assert round(float(d2(Xi[2], m1)), 5) == 1.51423
assert u((Fr(-13, 10), Fr(-43, 10))) == (Fr(1568, 100), Fr(45, 10))
assert round(float(d2(Xi[2], m2)), 5) == 4.29004
b = float(d2(Xi[2], m2) - d2(Xi[2], m1)) / 2
assert round(b, 5) == 1.38790 and round(1 / (1 + np.exp(-b)), 5) == 0.80026
R2, ll1 = tanggung_jawab(X, pi1, mu1, Sig1)
assert np.isclose(R2[0, 0], 1 / (1 + np.exp(-a))) and np.isclose(R2[2, 0], 1 / (1 + np.exp(-b)))
assert round(ll1, 4) == -26.7058

# Contoh Soal 4.3: kovarians sferis bersama
mu1v = (Fr(27, 10), Fr(27, 10))
sq = [(x[0] - mu1v[0]) ** 2 + (x[1] - mu1v[1]) ** 2 for x in Xi]
assert sq == [Fr(578, 100), Fr(458, 100), Fr(458, 100), Fr(2018, 100),
              Fr(2018, 100), Fr(3698, 100)]
t = sum(r * q for r, q in zip(r1, sq))
assert t == Fr(2586, 100)
assert Fr(9, 10) * Fr(578, 100) == Fr(5202, 1000)
assert Fr(3, 4) * Fr(916, 100) == Fr(687, 100)
assert Fr(1, 4) * Fr(4036, 100) == Fr(1009, 100)
assert Fr(1, 10) * Fr(3698, 100) == Fr(3698, 1000)
mu2v = (Fr(53, 10), Fr(53, 10))
t2 = sum(r * ((x[0] - mu2v[0]) ** 2 + (x[1] - mu2v[1]) ** 2) for r, x in zip(r2, Xi))
assert t2 == t and (t + t2) / 12 == Fr(431, 100) == S11

# teks
assert round(1 / (1 + np.exp(12)), 6) == 0.000006
print("Contoh Soal Bab 4: semua bilangan cocok")
