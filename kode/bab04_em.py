"""Bab 4: algoritma EM untuk GMM kovarians penuh, ditulis dari nol dan
dicocokkan iterasi demi iterasi dengan scikit-learn."""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from bab01_data import (BENIH, awal_mini, bangkitkan, data_mini, em,
                        langkah_m, tanggung_jawab)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()
pi0, mu0, S0 = awal_mini()

# (1) Satu iterasi: langkah E lalu langkah M
R, ll0 = tanggung_jawab(X, pi0, mu0, S0)
pi1, mu1, S1 = langkah_m(X, R)
R1, ll1 = tanggung_jawab(X, pi1, mu1, S1)
print("(1) Iterasi 1 dari awal mu = (3,3), (5,5), sigma0^2 = 6/log 3")
print("    r_i1 =", R[:, 0], " N =", R.sum(0))
print("    pi =", pi1, " mu1 =", mu1[0], " mu2 =", mu1[1])
print("    Sigma1 =", S1[0].round(4).tolist())
print("    Sigma2 =", S1[1].round(4).tolist())
print("    r_i1 baru =", R1[:, 0])
print("    log-kemungkinan: %.4f -> %.4f" % (ll0, ll1))

# (2) Riwayat log-kemungkinan dan pembanding scikit-learn
pi, mu, S, riw = em(X, pi0, mu0, S0, iterasi=500, tol=1e-8)
print("(2) Riwayat EM pada data mini:")
print("    t   log-kemungkinan   kenaikan")
for t in range(len(riw)):
    if t:
        print("   %2d   %14.6f   %10.6f" % (t, riw[t], riw[t] - riw[t - 1]))
    else:
        print("   %2d   %14.6f" % (t, riw[t]))
print("    berhenti sesudah %d iterasi" % (len(riw) - 1))
print("    akhir: pi =", pi, " mu1 = (%.7f, %.7f)" % tuple(mu[0]))
print("    r_i1 akhir =", " ".join("%.6f" % v for v in
                                tanggung_jawab(X, pi, mu, S)[0][:3, 0]),
      "...")
print("    Sigma1 =", S[0].round(4).tolist())
prec0 = np.array([np.linalg.inv(s) for s in S0])
for it in (1, 2, 500):
    gm = GaussianMixture(2, covariance_type="full", max_iter=it, tol=0,
                         reg_covar=0.0, weights_init=pi0, means_init=mu0,
                         precisions_init=prec0).fit(X)
    mu_ref = mu1 if it == 1 else (langkah_m(X, R1)[1] if it == 2 else mu)
    print("    sklearn max_iter = %3d: mu1 = %s  sama: %s"
          % (it, gm.means_[0].round(4), np.allclose(gm.means_, mu_ref,
                                                     atol=1e-8)))

# (3) Data lebih besar: 3 komponen, 900 sampel
rng = np.random.default_rng(BENIH)
pi_b = np.array([0.5, 0.3, 0.2])
mu_b = np.array([[0.0, 0.0], [5.0, 1.0], [2.0, 5.0]])
S_b = np.array([[[2.0, 0.8], [0.8, 1.0]], [[1.0, -0.6], [-0.6, 1.5]],
                [[0.5, 0.0], [0.0, 2.0]]])
Xb, zb = bangkitkan(900, pi_b, mu_b, S_b, rng)
awal = Xb[rng.choice(900, 3, replace=False)]
c = np.cov(Xb.T, bias=True)
pi_h, mu_h, S_h, riw_b = em(Xb, np.full(3, 1 / 3), awal,
                            np.array([c] * 3), iterasi=1000, tol=1e-10)
urut = np.argsort(-pi_h)
gm = GaussianMixture(3, covariance_type="full", tol=1e-10, max_iter=1000,
                     reg_covar=0.0, weights_init=np.full(3, 1 / 3),
                     means_init=awal,
                     precisions_init=np.array([np.linalg.inv(c)] * 3)).fit(Xb)
print("(3) 900 sampel dari 3 komponen; EM dari 3 sampel acak:")
print("    log-kemungkinan = %.3f, sklearn = %.3f"
      % (riw_b[-1], gm.score(Xb) * 900))
print("    pi benar =", pi_b)
print("    pi taksiran =", pi_h[urut])
for k in urut:
    print("    mu =", mu_h[k].round(3), " Sigma =", S_h[k].round(2).tolist())
