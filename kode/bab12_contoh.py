"""Memeriksa setiap bilangan Contoh Soal Bab 12."""
import numpy as np
from scipy.special import digamma

from bab01_data import awal_mini, data_mini, log_normal
from bab12_variasi import em_keras

X, g = data_mini()

# Contoh Soal 12.1: EM keras satu iterasi
pi, mu, S = em_keras(X, *awal_mini(), iterasi=1)
assert np.allclose(mu, [[2, 2], [6, 6]]) and np.allclose(S[0], [[2, -1], [-1, 2]])
pi2, mu2, S2 = em_keras(X, *awal_mini(), iterasi=5)
assert np.allclose(mu2, mu) and np.allclose(S2, S)
lc = 6 * np.log(0.5) + sum(log_normal(X[g == k], mu[k - 1], S[k - 1]).sum()
                           for k in (1, 2))
assert round(lc, 5) == -24.48198

# Contoh Soal 12.2: bobot u
u = lambda d2: 6 / (4 + d2)
assert [u(0), u(2), u(8), u(26)] == [1.5, 1.0, 0.5, 0.2]
Si = np.array([[2, 1], [1, 2]]) / 3
assert np.isclose(np.array([-2, -5]) @ Si @ np.array([-2, -5]), 26)
assert np.isclose(np.array([2, 2]) @ Si @ np.array([2, 2]), 8)

# Contoh Soal 12.3: digamma
assert round(digamma(1), 4) == -0.5772 and round(np.exp(digamma(1)), 4) == 0.5615
assert round(digamma(2), 4) == 0.4228 and round(np.exp(digamma(2)), 4) == 1.5262
assert round(sum(1 / j for j in range(1, 10)), 4) == 2.8290
assert round(digamma(10), 4) == 2.2518 and round(np.exp(digamma(10)), 3) == 9.504
assert round(np.exp(digamma(2)) / 2, 2) == 0.76 and round(np.exp(digamma(10)) / 10, 2) == 0.95
print("Contoh Soal Bab 12: semua bilangan cocok")
