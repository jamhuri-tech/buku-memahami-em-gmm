"""Memeriksa setiap bilangan Contoh Soal Bab 2."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, log_normal

X, g = data_mini()
Xi = X.astype(int)
m, n = X.shape

# Contoh Soal 2.1
D = Xi - 4
assert [tuple(v) for v in D] == [(-3, -3), (-3, 0), (0, -3), (0, 3), (3, 0), (3, 3)]
S = [[Fr(int((D[:, a] * D[:, b]).sum()), 6) for b in range(2)] for a in range(2)]
assert S == [[6, 3], [3, 6]]
det = S[0][0] * S[1][1] - S[0][1] ** 2
assert det == 27
Si = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]]
assert Si == [[Fr(2, 9), Fr(-1, 9)], [Fr(-1, 9), Fr(2, 9)]]
val, vec = np.linalg.eigh(np.array(S, float))
assert np.allclose(val, [3, 9])
assert np.allclose(np.abs(vec[:, 1]), [1 / np.sqrt(2)] * 2)
assert np.isclose(vec[0, 0] * vec[1, 0], -0.5)
d2 = [Fr(2 * a * a - 2 * a * b + 2 * b * b, 9) for a, b in D]
assert d2 == [2] * 6 and sum(d2) == m * n

# Contoh Soal 2.2
ll = -3 * (2 * np.log(2 * np.pi) + np.log(27) + 2)
assert round(2 * np.log(2 * np.pi), 5) == 3.67575
assert round(np.log(27), 5) == 3.29584 and round(ll, 4) == -26.9148
assert np.isclose(ll, log_normal(X, X.mean(0), np.array(S, float)).sum())
A = Xi[g == 1] - 2
assert [tuple(v) for v in A] == [(-1, -1), (-1, 2), (2, -1)]
assert (A[:, 0] ** 2).sum() == 6 and (A[:, 1] ** 2).sum() == 6
assert (A[:, 0] * A[:, 1]).sum() == -3
S1 = np.array([[2, -1], [-1, 2]])
assert np.isclose(np.linalg.det(S1), 3)
assert np.allclose(np.linalg.eigvalsh(S1), [1, 3])
assert np.allclose(S1 @ [1, -1], [3, -3]) and np.allclose(S1 @ [1, 1], [1, 1])
assert np.isclose(-1 / np.sqrt(4), -0.5) and np.isclose(3 / np.sqrt(36), 0.5)

# Contoh Soal 2.3: x2 | x1 = 7
assert 4 + Fr(3, 6) * (7 - 4) == Fr(11, 2) and 6 - Fr(9, 6) == Fr(9, 2)
assert 6 + Fr(-1, 2) * (7 - 6) == Fr(11, 2) and 2 - Fr(1, 2) == Fr(3, 2)
assert Fr(9, 2) / Fr(3, 2) == 3

# teks: batas 95% chi2_2 dan bias
assert round(-2 * np.log(0.05), 4) == 5.9915
from scipy import stats
assert np.isclose(stats.chi2.ppf(0.95, 2), -2 * np.log(0.05))
assert Fr(5, 6) == Fr(6 - 1, 6)
print("Contoh Soal Bab 2: semua bilangan cocok")
