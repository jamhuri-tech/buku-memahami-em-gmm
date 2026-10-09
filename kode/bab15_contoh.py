"""Memeriksa setiap bilangan Contoh Soal Bab 15."""
from math import comb, log

import numpy as np

# Contoh Soal 15.1
assert 2 + 12 + 10 == 24 and 3 + 16 + 10 == 29 and 4 * 5 // 2 == 10
assert round(log(342), 4) == 5.8348 and round(24 * log(342), 2) == 140.04
assert round(-(2515.8 - 24 * log(342)) / 2, 1) == -1187.9
assert round(2515.8 - 2501.4, 1) == 14.4

# Contoh Soal 15.2
e3, e4 = (2532.4 - 2515.8) / 2, (2593.3 - 2501.4) / 2
assert round(e3, 1) == 8.3 and round(e4, 1) == 46.0
assert round(2 * (round(e4, 1) - round(e3, 1)), 1) == 75.4

# Contoh Soal 15.3: ARI
sel = comb(150, 2) + comb(4, 2) + comb(64, 2) + comb(123, 2)
assert (comb(150, 2), comb(64, 2), comb(123, 2), sel) == (11175, 2016, 7503, 20700)
a = comb(151, 2) + comb(68, 2) + comb(123, 2)
b = comb(123, 2) + comb(154, 2) + comb(65, 2)
assert (a, b, comb(342, 2), comb(154, 2), comb(65, 2)) == (21106, 21364, 58311, 11781, 2080)
E = a * b / comb(342, 2)
assert round(E, 1) == 7732.8
ari = (sel - E) / ((a + b) / 2 - E)
assert round(sel - E, 1) == 12967.2 and round((a + b) / 2 - E, 1) == 13502.2
assert round(ari, 3) == 0.960

# teks: bagian spesies
assert [round(v / 342, 3) for v in (151, 68, 123)] == [0.442, 0.199, 0.360]
print("Contoh Soal Bab 15: semua bilangan cocok")
