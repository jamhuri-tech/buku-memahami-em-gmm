"""Memeriksa setiap bilangan Contoh Soal Bab 14."""
import numpy as np

# Contoh Soal 14.1
assert round(1 / np.sqrt(9.12), 4) == 0.3311 and round(1 / np.sqrt(24), 4) == 0.2041
assert np.isclose(1 / np.sqrt(24), np.sqrt(0.25 / 6)) and round(24 / 9.12, 2) == 2.63

# Contoh Soal 14.2
assert round(np.sqrt(2 / 3), 4) == 0.8165 and round(1.96 * np.sqrt(2 / 3), 2) == 1.60
assert round(2 - 1.96 * np.sqrt(2 / 3), 2) == 0.40 and round(2 + 1.96 * np.sqrt(2 / 3), 2) == 3.60

# Contoh Soal 14.3
assert round(0.223 * 0.777 * 0.225, 3) == 0.039
lo, hi = -1.247 - 1.96 * 0.225, -1.247 + 1.96 * 0.225
assert round(lo, 3) == -1.688 and round(hi, 3) == -0.806
assert round(np.exp(-1.688), 4) == 0.1849 and round(np.exp(-0.806), 4) == 0.4466
s = lambda e: 1 / (1 + np.exp(-e))
assert round(s(lo), 3) == 0.156 and round(s(hi), 3) == 0.309
print("Contoh Soal Bab 14: semua bilangan cocok")
