"""Memeriksa setiap bilangan Contoh Soal Bab 11."""
from fractions import Fraction as Fr
from math import comb

import numpy as np
from sklearn.metrics import adjusted_rand_score

from bab01_data import akhir_mini, data_mini, log_gabung, log_normal, logsumexp

X, g = data_mini()

# Contoh Soal 11.1: ARI
n = [[2, 1], [1, 2]]
sab = sum(comb(v, 2) for r in n for v in r)
sa = sum(comb(sum(r), 2) for r in n)
sb = sum(comb(sum(c), 2) for c in zip(*n))
E = Fr(sa * sb, comb(6, 2))
ari = (sab - E) / (Fr(sa + sb, 2) - E)
assert (sab, sa, sb, comb(6, 2), E) == (2, 6, 6, 15, Fr(12, 5))
assert ari == Fr(-1, 9)
assert np.isclose(adjusted_rand_score(g, [1, 1, 2, 1, 2, 2]), -1 / 9)

# Contoh Soal 11.2: skor anomali
mu1, S1 = X.mean(0), np.cov(X.T, bias=True)
a_g = -log_normal(np.array([[4.0, 4.0]]), mu1, S1)[0]
assert round(np.log(2 * np.pi), 5) == 1.83788 and round(0.5 * np.log(27), 5) == 1.64792
assert round(a_g, 4) == 3.4858 and round(a_g + 1, 4) == 4.4858
Si = np.linalg.inv(akhir_mini()[2][0])
assert np.allclose(Si, np.array([[2, 1], [1, 2]]) / 3)
assert np.isclose(np.array([2, 2]) @ Si @ np.array([2, 2]), 8)
assert np.isclose(np.array([-1, 2]) @ Si @ np.array([-1, 2]), 2)
a_m = -logsumexp(log_gabung(np.array([[4.0, 4.0]]), *akhir_mini()))[0]
assert round(0.5 * np.log(3), 5) == 0.54931
assert round(a_m, 4) == 6.3872 and np.isclose(a_m, 4 + np.log(2 * np.pi) + 0.5 * np.log(3))
lat = -logsumexp(log_gabung(X, *akhir_mini()))
assert round(lat.max(), 4) == 4.0803
assert round(np.log(2) + 1 + np.log(2 * np.pi) + 0.5 * np.log(3), 4) == 4.0803

# Contoh Soal 11.3: GMM per kelas lawan QDA
f = lambda x, m: np.exp(-(x - m) ** 2)
p0 = lambda x: (f(x, 0) + f(x, 4)) / (f(x, 0) + f(x, 4) + f(x, 2) + f(x, 6))
assert round(np.exp(2) / (1 + np.exp(2)), 4) == 0.8808 and round(p0(0.5), 4) == 0.8808
assert round(p0(4.5), 4) == 0.8789
assert 0.5 + 0.5 * (4 + 4) == 4.5
lo = lambda x: ((x - 4) ** 2 - (x - 2) ** 2) / 9
assert np.isclose(lo(0.5), 10 / 9) and np.isclose(lo(4.5), -2 / 3)
assert all(np.isclose(lo(x), (12 - 4 * x) / 9) for x in (0, 1, 3, 7))
assert round(1 / (1 + np.exp(-10 / 9)), 4) == 0.7523
assert round(1 / (1 + np.exp(2 / 3)), 4) == 0.3392
print("Contoh Soal Bab 11: semua bilangan cocok")
