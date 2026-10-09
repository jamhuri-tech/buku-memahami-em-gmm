"""Memeriksa setiap bilangan Contoh Soal Bab 5."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import (awal_mini, data_mini, langkah_m, log_gabung,
                        tanggung_jawab)

X, g = data_mini()
pi0, mu0, S0 = awal_mini()

# Contoh Soal 5.1: KL satu sampel
kl = 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)
assert round(np.log(0.5 / 0.9), 5) == -0.58779 and round(np.log(5), 5) == 1.60944
assert round(kl, 5) == 0.51083
A1 = log_gabung(X[:1], pi0, mu0, S0)[0]
lp1 = np.log(np.exp(A1).sum())
F1 = 0.5 * (A1 - np.log(0.5)).sum()
assert np.isclose(lp1 - F1, kl) and round(F1, 4) == -5.3666
assert round(lp1, 4) == -4.8558

# Contoh Soal 5.2: entropi, Q, rantai
h1 = -0.9 * np.log(0.9) - 0.1 * np.log(0.1)
h2 = -0.75 * np.log(0.75) - 0.25 * np.log(0.25)
assert round(-0.9 * np.log(0.9), 5) == 0.09482 and round(-0.1 * np.log(0.1), 5) == 0.23026
assert round(h1, 5) == 0.32508 and round(h2, 5) == 0.56234
assert round(-0.75 * np.log(0.75), 5) == 0.21576 and round(-0.25 * np.log(0.25), 5) == 0.34657
H = 2 * h1 + 4 * h2
assert round(H, 5) == 2.89951
R, ll0 = tanggung_jawab(X, pi0, mu0, S0)
assert round(ll0, 4) == -27.3068 and round(ll0 - H, 4) == -30.2063
assert np.isclose((R * log_gabung(X, pi0, mu0, S0)).sum(), ll0 - H)
assert Fr(431, 100) ** 2 - Fr(131, 100) ** 2 == Fr(1686, 100)
assert Fr(431, 100) ** 2 == Fr(185761, 10000) and Fr(131, 100) ** 2 == Fr(17161, 10000)
Q1 = 6 * np.log(0.5) - 3 * (2 * np.log(2 * np.pi) + np.log(16.86) + 2)
assert round(6 * np.log(0.5), 5) == -4.15888 and round(np.log(16.86), 5) == 2.82494
assert round(3 * (2 * np.log(2 * np.pi) + np.log(16.86) + 2), 5) == 25.50209
assert round(Q1, 5) == -29.66098
pi1, mu1, S1 = langkah_m(X, R)
assert np.isclose((R * log_gabung(X, pi1, mu1, S1)).sum(), Q1)
ll1 = tanggung_jawab(X, pi1, mu1, S1)[1]
assert round(Q1 + H, 4) == -26.7615 and round(ll1, 4) == -26.7058
assert round(Q1 - (ll0 - H), 4) == 0.5453 and round(ll1 - (Q1 + H), 4) == 0.0557
assert ll1 >= Q1 + H >= ll0

# Contoh Soal 5.3: EM untuk pi saja
rho = [Fr(9), Fr(3), Fr(3), Fr(1, 3), Fr(1, 3), Fr(1, 9)]
assert np.allclose(np.exp(log_gabung(X, np.array([.5, .5]), mu0, S0) @ [1, -1]),
                   [float(v) for v in rho])
p = Fr(3, 4)
r = [p * v / (p * v + 1 - p) for v in rho]
assert r == [Fr(27, 28), Fr(9, 10), Fr(9, 10), Fr(1, 2), Fr(1, 2), Fr(1, 4)]
assert sum(r) == Fr(562, 140) and sum(r) / 6 == Fr(281, 420)
assert round(float(Fr(281, 420)), 6) == 0.669048
rat = [(Fr(3, 4) * v + Fr(1, 4)) / (Fr(1, 2) * v + Fr(1, 2)) for v in rho]
assert rat == [Fr(7, 5), Fr(5, 4), Fr(5, 4), Fr(3, 4), Fr(3, 4), Fr(3, 5)]
prod = np.prod([float(v) for v in rat])
assert np.prod(rat) == Fr(4725, 6400) and round(np.log(prod), 5) == -0.30343
assert 7 * 25 * 9 * 3 == 4725 and 5 * 16 * 16 * 5 == 6400
# teks: jarak ke 1/2 menyusut
d = [0.25, 0.669048 - 0.5, 0.608897 - 0.5, 0.568579 - 0.5]
assert [round(v, 3) for v in d] == [0.25, 0.169, 0.109, 0.069]
print("Contoh Soal Bab 5: semua bilangan cocok")
