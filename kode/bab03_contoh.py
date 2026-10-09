"""Memeriksa setiap bilangan Contoh Soal Bab 3."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import akhir_mini, data_mini, log_normal, tanggung_jawab

X, g = data_mini()
pi, mu, S = akhir_mini()

# Contoh Soal 3.1: batas keputusan
Si = [[Fr(2, 3), Fr(1, 3)], [Fr(1, 3), Fr(2, 3)]]
assert np.allclose(np.array(Si, float), np.linalg.inv(S[0]))
d = (-4, -4)
w = tuple(Si[a][0] * d[0] + Si[a][1] * d[1] for a in range(2))
assert w == (-4, -4)
q = lambda v: sum(v[a] * Si[a][b] * v[b] for a in range(2) for b in range(2))
assert q((2, 2)) == 8 and q((6, 6)) == 72
w0 = -Fr(1, 2) * (q((2, 2)) - q((6, 6)))
assert w0 == 32
lo = lambda x: w[0] * x[0] + w[1] * x[1] + w0
assert lo((4, 1)) == 12 and lo((4, 4)) == 0 and lo((5, 4)) == -4
R, _ = tanggung_jawab(np.array([[4, 1], [4, 4], [5, 4]], float), pi, mu, S)
assert round(R[0, 0], 6) == 0.999994 and np.isclose(R[1, 0], 0.5)
assert round(R[2, 0], 6) == 0.017986
assert np.isclose(R[2, 0], 1 / (1 + np.exp(4)))

# Contoh Soal 3.2: hukum varians total
mbar = 0.5 * mu[0] + 0.5 * mu[1]
assert np.allclose(mbar, [4, 4])
W = 0.5 * S[0] + 0.5 * S[1]
B = 0.5 * np.outer(mu[0] - mbar, mu[0] - mbar) + 0.5 * np.outer(mu[1] - mbar, mu[1] - mbar)
assert np.allclose(B, [[4, 4], [4, 4]]) and np.linalg.matrix_rank(B) == 1
assert np.allclose(W + B, [[6, 3], [3, 6]])
assert np.allclose(W + B, np.cov(X.T, bias=True))

# Contoh Soal 3.3: singularitas
c = np.exp(-1) / (2 * np.pi * np.sqrt(27))
assert round(c, 6) == 0.011268
mu_s, S_s = X.mean(0), np.cov(X.T, bias=True)
assert np.allclose(np.exp(log_normal(X, mu_s, S_s)), c)
t = np.log(0.5 * c)
assert round(t, 5) == -5.17894 and round(5 * t, 5) == -25.89471
assert round(np.log(0.5), 5) == -0.69315 and round(np.log(2 * np.pi), 5) == 1.83788
assert round(0.5 * np.log(27), 5) == 1.64792
a = 1 / (2 * np.pi * 0.01)
assert round(a, 5) == 15.91549 and round(0.5 * (a + c), 5) == 7.96338
s1 = np.log(0.5 * (a + c))
assert round(s1, 5) == 2.07485 and round(s1 + 5 * t, 5) == -23.81986
_, ll = tanggung_jawab(X, pi, np.array([X[0], mu_s]),
                       np.array([0.01 * np.eye(2), S_s]))
assert round(ll, 4) == -23.8199
T = round(-24.4820 + 25.8947, 4)
assert T == 1.4127 and round(np.exp(T), 4) == 4.1070
eps = 1 / (2 * np.pi * (2 * np.exp(T) - c))
assert round(2 * np.exp(T) - c, 4) == 8.2028 and round(eps, 4) == 0.0194
# batas sebenarnya dari kode
from scipy.optimize import brentq
f = lambda e: tanggung_jawab(X, pi, np.array([X[0], mu_s]),
                             np.array([e * np.eye(2), S_s]))[1] + 24.481958
assert round(brentq(f, 1e-3, 0.1), 4) == 0.0194
assert round(np.sqrt(0.0194), 2) == 0.14
assert round(np.log(10), 4) == 2.3026
print("Contoh Soal Bab 3: semua bilangan cocok")
