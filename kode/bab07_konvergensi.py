"""Bab 7: laju konvergensi EM, jebakan simetri, dan strategi pemulaan."""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from bab01_data import (BENIH, SIGMA0_2, awal_mini, bangkitkan, data_mini,
                        em, log_gabung, tanggung_jawab)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()
_, mu0, S0 = awal_mini()

# (1) Laju linear model satu parameter (pi saja)
rho = np.exp(log_gabung(X, np.array([.5, .5]), mu0, S0) @ [1, -1])
M = lambda p: np.mean(p * rho / (p * rho + 1 - p))
Ic = 6 / 0.25
Iobs = np.sum(((rho - 1) / (0.5 * rho + 0.5)) ** 2)
p, jarak = 0.75, []
for t in range(12):
    jarak.append(p - 0.5)
    p = M(p)
print("(1) EM pi saja dari 3/4: rasio jarak berturut-turut")
print("   ", " ".join("%.4f" % (jarak[t + 1] / jarak[t]) for t in range(0, 11, 2)))
print("    I_c = %.2f  I_obs = %.2f  1 - I_obs/I_c = %.4f"
      % (Ic, Iobs, 1 - Iobs / Ic))
h = 1e-6
print("    M'(1/2) beda hingga = %.4f" % ((M(0.5 + h) - M(0.5 - h)) / (2 * h)))

# (2) Jebakan simetri
S_awal = np.array([SIGMA0_2 * np.eye(2)] * 2)
pa = np.array([.5, .5])
print("(2) Jebakan simetri; awal Sigma = sigma0^2 I, pi = (1/2, 1/2)")
for e in (0.0, 0.01, -0.01):
    mu_a = np.array([[3.0, 4.0 + e], [5.0, 4.0]])
    pi, mu, S, riw = em(X, pa, mu_a, S_awal, iterasi=3000, tol=1e-12)
    R, ll = tanggung_jawab(X, pi, mu, S)
    print("    awal mu1 = (3, %.2f): l akhir = %.4f" % (4 + e, ll))
    print("      mu1 akhir =", mu[0].round(3))
    print("      r_i1 =", " ".join("%.4f" % v for v in R[:, 0]))

# (3) Strategi pemulaan scikit-learn, 200 benih, n_init = 1
rng = np.random.default_rng(BENIH)
pi_b = np.array([0.25] * 4)
mu_b = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], float)
S_b = np.array([np.eye(2) * 1.2] * 4)
Xb, _ = bangkitkan(400, pi_b, mu_b, S_b, rng)
terbaik = max(GaussianMixture(4, n_init=20, random_state=s).fit(Xb).score(Xb)
              for s in range(5)) * 400
print("(3) 400 sampel, 4 komponen; l terbaik = %.2f" % terbaik)
print("    %-17s capai terbaik   rata-rata l   iterasi" % "init_params")
for cara in ("random", "random_from_data", "k-means++", "kmeans"):
    ll, it = [], []
    for s in range(200):
        gm = GaussianMixture(4, init_params=cara, n_init=1, max_iter=500,
                             random_state=s).fit(Xb)
        ll.append(gm.score(Xb) * 400)
        it.append(gm.n_iter_)
    ll = np.array(ll)
    print("    %-17s %5.1f%%        %9.2f   %6.1f"
          % (cara, 100 * np.mean(ll > terbaik - 0.5), ll.mean(), np.mean(it)))
