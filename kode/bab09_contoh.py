"""Memeriksa setiap bilangan Contoh Soal Bab 9."""
import numpy as np

# Contoh Soal 9.1
l1, l2 = -26.914773, -24.481958
assert round(np.log(6), 5) == 1.79176
assert round(-2 * l1, 4) == 53.8295 and round(-2 * l2, 4) == 48.9639
assert round(5 * np.log(6), 4) == 8.9588 and round(11 * np.log(6), 4) == 19.7094
assert round(-2 * l1 + 10, 3) == 63.830 and round(-2 * l1 + 5 * np.log(6), 3) == 62.788
assert round(-2 * l2 + 22, 3) == 70.964 and round(-2 * l2 + 11 * np.log(6), 3) == 68.673
assert round(l2 - l1, 2) == 2.43 and round(6 * np.log(6), 2) == 10.75
assert round(3 * np.log(6), 2) == 5.38

# Contoh Soal 9.2: data mini diulang 10 kali
assert round(np.log(60), 5) == 4.09434
b1 = -20 * l1 + 5 * np.log(60)
b2 = -20 * l2 + 11 * np.log(60)
assert round(-20 * l1, 3) == 538.295 or round(-20 * l1, 3) == 538.296
assert round(5 * np.log(60), 3) == 20.472 and round(11 * np.log(60), 3) == 45.038
assert round(b1, 3) == 558.767 or round(b1, 3) == 558.768
assert round(b2, 3) == 534.677 and round(b1 - b2) == 24
# MLE data ulang sama dengan MLE data mini
from sklearn.mixture import GaussianMixture
from bab01_data import data_mini
X, g = data_mini()
X10 = np.tile(X, (10, 1))
gm = GaussianMixture(2, reg_covar=1e-9, tol=1e-12, max_iter=2000,
                     means_init=[[2, 2], [6, 6]]).fit(X10)
assert np.isclose(gm.score(X10) * 60, 10 * l2, atol=1e-3)

# Contoh Soal 9.3: penalti entropi
H = lambda p: -p * np.log(p) - (1 - p) * np.log(1 - p)
assert round(-0.999 * np.log(0.999), 6) == 0.000999
assert round(-0.001 * np.log(0.001), 6) == 0.006908
assert round(H(0.999), 6) == 0.007907 and round(800 * H(0.999), 2) == 6.33
assert round(-0.6 * np.log(0.6), 5) == 0.30650 and round(-0.4 * np.log(0.4), 5) == 0.36652
assert round(H(0.6), 5) == 0.67301 and round(800 * H(0.6), 2) == 538.41
assert round(6 * np.log(400), 1) == 35.9
print("Contoh Soal Bab 9: semua bilangan cocok")
