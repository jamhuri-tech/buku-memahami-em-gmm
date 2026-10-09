"""Memeriksa setiap bilangan Contoh Soal Bab 8."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, log_normal

X, g = data_mini()

# Contoh Soal 8.1
S1 = np.array([[2.0, -1.0], [-1.0, 2.0]])
assert np.allclose(0.5 * S1 + 0.5 * S1, S1)
assert np.allclose(np.diag(np.diag(S1)), 2 * np.eye(2)) and np.trace(S1) / 2 == 2
D = X[g == 1] - 2
assert list((D ** 2).sum(1)) == [2, 5, 5] and (D ** 2).sum() / 2 == 6
mu = {1: np.array([2., 2.]), 2: np.array([6., 6.])}
llp = 6 * np.log(0.5) + sum(log_normal(X[g == k], mu[k], S1).sum() for k in (1, 2))
lld = 6 * np.log(0.5) + sum(log_normal(X[g == k], mu[k], 2 * np.eye(2)).sum() for k in (1, 2))
assert round(6 * np.log(2 * np.pi), 5) == 11.02726 and round(3 * np.log(3), 5) == 3.29584
assert round(llp, 5) == -24.48198 and round(lld, 5) == -25.34503
assert np.isclose(lld - llp, -3 * np.log(4 / 3)) and round(3 * np.log(4 / 3), 5) == 0.86305

# Contoh Soal 8.2: banyaknya parameter
def param(K, n, j):
    kov = {"full": K * n * (n + 1) // 2, "tied": n * (n + 1) // 2,
           "diag": K * n, "spherical": K}[j]
    return K - 1 + K * n + kov
assert [param(5, 10, j) for j in ("full", "tied", "diag", "spherical")] == [329, 109, 104, 59]
assert [param(2, 2, j) for j in ("full", "tied", "diag", "spherical")] == [11, 8, 9, 7]
assert 1 / 7 < 329 / 2000 and 59 / 2000 < 1 / 30

# Contoh Soal 8.3: limit K-Means
assert round(1 / (1 + np.exp(-12 * np.log(3) / 12)), 6) == 0.75
assert round(1 / (1 + np.exp(-6)), 6) == 0.997527
d = ((X[:, None, :] - np.array([[3, 3], [5, 5]])[None]) ** 2).sum(-1)
assert list(d[:, 1] - d[:, 0] > 0) == [True] * 3 + [False] * 3
assert np.allclose(d[:, 1] - d[:, 0], -4 * (X.sum(1) - 8))
iner = sum(((X[g == k] - mu[k]) ** 2).sum() for k in (1, 2))
assert iner == 24
assert round(5241.0 - 4743.8) == 497
print("Contoh Soal Bab 8: semua bilangan cocok")
