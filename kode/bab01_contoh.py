"""Memeriksa setiap bilangan Contoh Soal Bab 1."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import awal_mini, data_mini, tanggung_jawab, log_normal

X, g = data_mini()
Xi = X.astype(int)

# Contoh Soal 1.1: rata-rata kolom dan kelompok
assert Xi[:, 0].sum() == 24 and Xi[:, 1].sum() == 24
assert np.allclose(X.mean(0), [4, 4])
assert np.allclose(X[g == 1].mean(0), [2, 2])
assert np.allclose(X[g == 2].mean(0), [6, 6])
# simetri cermin terhadap (4, 4)
assert sorted(map(tuple, 8 - Xi[g == 1])) == sorted(map(tuple, Xi[g == 2]))

# Contoh Soal 1.2: tanggung jawab dengan pangkat 3
pi, mu, S = awal_mini()
d2 = ((X[:, None, :] - mu[None]) ** 2).sum(-1)
assert np.allclose(d2[0], [8, 32]) and np.allclose(d2[2], [5, 17])
assert np.allclose(d2[:, 0] - d2[:, 1], 4 * (X.sum(1) - 8))
r = [1 / (1 + Fr(3) ** int((d2[i, 0] - d2[i, 1]) // 12)) for i in range(6)]
assert r == [Fr(9, 10), Fr(3, 4), Fr(3, 4), Fr(1, 4), Fr(1, 4), Fr(1, 10)]
R, ll = tanggung_jawab(X, pi, mu, S)
assert np.allclose(R[:, 0], [float(v) for v in r])
# faktor depan dan p(x_1)
assert np.isclose(1 / (2 * np.pi * S[0, 0, 0]), np.log(3) / (12 * np.pi))
c = np.log(3) / (24 * np.pi)
assert round(c, 6) == 0.014571
assert round(3 ** (-2 / 3), 5) == 0.48075 and round(3 ** (-8 / 3), 5) == 0.05342
assert round(3 ** (-2 / 3) + 3 ** (-8 / 3), 5) == 0.53417
p1 = c * (3 ** (-2 / 3) + 3 ** (-8 / 3))
assert round(p1, 7) == 0.0077832
assert round(np.log(p1), 4) == -4.8558
assert np.isclose(np.log(p1), np.log(0.5 * np.exp(log_normal(X[:1], mu[0], S[0]))
                                   + 0.5 * np.exp(log_normal(X[:1], mu[1], S[1]))))
assert np.isclose(3 ** (-8 / 3) / (3 ** (-2 / 3) + 3 ** (-8 / 3)), 0.1)
assert round(ll, 2) == -27.31

# Contoh Soal 1.3: turunan dan rata-rata berbobot
gmu = 0.9 * (np.log(3) / 6) * (X[0] - mu[0])
assert np.allclose(gmu, -3 * np.log(3) / 10) and round(gmu[0], 4) == -0.3296
assert Fr(9, 10) - Fr(1, 2) == Fr(2, 5)
N1 = sum(r)
assert N1 == 3
s1 = sum(ri * int(x) for ri, x in zip(r, Xi[:, 0]))
s2 = sum(ri * int(x) for ri, x in zip(r, Xi[:, 1]))
assert s1 == s2 == Fr(81, 10)
assert s1 / N1 == Fr(27, 10) and N1 / 6 == Fr(1, 2)

# teks: banyaknya parameter K = 2, n = 2 dan selisih log-likelihood
K, n = 2, 2
assert (K - 1) + K * n + K * n * (n + 1) // 2 == 11
assert round(-24.4820 + 26.9148, 2) == 2.43 and round(np.exp(2.43)) == 11
assert np.isclose(6 / np.log(3), 5.4614, atol=1e-4)
print("Contoh Soal Bab 1: semua bilangan cocok")
