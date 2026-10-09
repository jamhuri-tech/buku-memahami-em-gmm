"""Memeriksa setiap bilangan Contoh Soal Bab 6."""
from fractions import Fraction as Fr

import numpy as np

from bab06_model import B, bern_e, bern_m

# Contoh Soal 6.1: campuran Bernoulli
Bi = B.astype(int)
s = Bi.sum(1)
assert list(s) == [0, 1, 1, 3, 2, 2]
r = [1 / (1 + Fr(3) ** (3 - 2 * int(v))) for v in s]
assert r == [Fr(1, 28), Fr(1, 4), Fr(1, 4), Fr(27, 28), Fr(3, 4), Fr(3, 4)]
for v in range(4):
    rasio = (Fr(3, 4) ** v * Fr(1, 4) ** (3 - v)) / (Fr(1, 4) ** v * Fr(3, 4) ** (3 - v))
    assert rasio == Fr(3) ** (2 * v - 3)
N1 = sum(r)
assert N1 == 3 and (1 + 7 + 7 + 27 + 21 + 21) == 84
th1 = [sum(ri * int(x) for ri, x in zip(r, Bi[:, j])) / N1 for j in range(3)]
assert th1 == [Fr(55, 84), Fr(23, 28), Fr(55, 84)]
assert Fr(1, 4) + Fr(27, 28) + Fr(3, 4) == Fr(55, 28)
assert round(float(th1[0]), 4) == 0.6548 and round(float(th1[1]), 4) == 0.8214
R, _ = bern_e(B, np.array([.5, .5]), np.array([[.75] * 3, [.25] * 3]))
_, th = bern_m(B, R)
assert np.allclose(th[0], [float(v) for v in th1])
assert np.allclose(th[1], [29 / 84, 5 / 28, 29 / 84])
# limit
lim = 6 * np.log(0.5) + 2 * np.log(4 / 9) + 4 * np.log(2 / 9)
assert round(lim, 6) == -11.797053
assert np.isclose(bern_e(B, np.array([.5, .5]),
                         np.array([[2 / 3, 1, 2 / 3], [1 / 3, 0, 1 / 3]]))[1], lim)

# Contoh Soal 6.2: data hilang
x1 = [1, 1, 4, 4, 7, 7]
x2 = [Fr(1), Fr(4), Fr(1), None, Fr(4), Fr(7)]
o = [i for i in range(6) if x2[i] is not None]
m1 = Fr(sum(x1[i] for i in o), 5)
m2 = Fr(sum(x2[i] for i in o), 5)
assert (m1, m2) == (4, Fr(17, 5))
s11 = sum((x1[i] - m1) ** 2 for i in o) / 5
s12 = sum((x1[i] - m1) * (x2[i] - m2) for i in o) / 5
s22 = sum((x2[i] - m2) ** 2 for i in o) / 5
assert (s11, s12, s22) == (Fr(36, 5), Fr(18, 5), Fr(126, 25))
xh = m2 + s12 / s11 * (4 - m1)
v = s22 - s12 ** 2 / s11
assert xh == Fr(17, 5) and v == Fr(81, 25) and s12 ** 2 / s11 == Fr(9, 5)
z2 = [x2[i] if x2[i] is not None else xh for i in range(6)]
M2 = sum(z2) / 6
assert sum(z2) == Fr(102, 5) and M2 == Fr(17, 5)
S11 = Fr(sum((a - 4) ** 2 for a in x1), 6)
S12 = sum((a - 4) * (b - M2) for a, b in zip(x1, z2)) / 6
S22 = (sum((b - M2) ** 2 for b in z2) + v) / 6
assert (S11, S12, S22) == (6, 3, Fr(237, 50))
assert sum((b - M2) ** 2 for b in z2) + v == Fr(2844, 100)
assert M2 + S12 / S11 * 0 == Fr(17, 5) and S22 - S12 ** 2 / S11 == Fr(81, 25)
b1 = s12 / s11
assert b1 == Fr(1, 2) and m2 - b1 * 4 == Fr(7, 5) and Fr(7, 5) + b1 * 4 == Fr(17, 5)

# Contoh Soal 6.3: isian GMM
assert 2 + Fr(-1, 2) * (4 - 2) == 1 and 6 + Fr(-1, 2) * (4 - 6) == 7
assert 2 - Fr(1, 2) == Fr(3, 2)
assert Fr(3, 2) + Fr(1, 2) * 9 + Fr(1, 2) * 9 == Fr(21, 2)
print("Contoh Soal Bab 6: semua bilangan cocok")
