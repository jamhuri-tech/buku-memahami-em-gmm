"""Memeriksa setiap bilangan Contoh Soal Bab 10."""
from fractions import Fraction as Fr

import numpy as np

from bab10_singular import X7, em_map, pi0, mu0, S0

# Contoh Soal 10.1
for eps, nilai in ((1e-2, 1.9535), (1e-6, 6.5612)):
    det = (2 + eps) * eps
    assert round(-0.5 * np.log(det), 4) == nilai
assert np.isclose((2.01) * 0.01, 0.0201)
assert round(0.5 * np.log(1e4), 2) == 4.61

# Contoh Soal 10.2: EM-MAP
pi, mu, S, _ = em_map(X7, pi0, mu0, S0, alpha=2.0, Psi=0.5 * np.eye(2),
                      nu=1.0, iterasi=500)
N = [2, 2, 3]
assert [Fr(k + 1, 10) for k in N] == [Fr(3, 10), Fr(3, 10), Fr(4, 10)]
assert np.allclose(pi, [0.3, 0.3, 0.4])
assert np.allclose(S[0], np.eye(2) / 12, atol=1e-6)
assert np.allclose(mu[1], [2.5, 2.5], atol=1e-6)
assert np.allclose(S[1], np.array([[5, -4.5], [-4.5, 5]]) / 6, atol=1e-6)
assert np.allclose(S[2], np.array([[6.5, -3], [-3, 6.5]]) / 7, atol=1e-6)
assert round(5 / 6, 3) == 0.833 and round(6.5 / 7, 3) == 0.929
assert round(3 / 7, 3) == 0.429

# Contoh Soal 10.3: varians MAP untuk satu titik
psi = 0.5
f = lambda lam: -5 * np.log(lam) - psi / lam
lam = np.linspace(0.01, 1, 100000)
assert np.isclose(lam[np.argmax(f(lam))], psi / 5, atol=1e-4)
assert psi / (1 + 1 + 2 + 1) == 0.1
# reg_covar sebagai kasus khusus Wishart terbalik
Nk, eps, Sk = 3.0, 0.1, np.diag([2.0, 0.0])
assert np.allclose((Nk * Sk + Nk * eps * np.eye(2)) / Nk, Sk + eps * np.eye(2))
print("Contoh Soal Bab 10: semua bilangan cocok")
