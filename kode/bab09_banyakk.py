"""Bab 9: memilih banyaknya komponen dengan AIC, BIC, ICL, uji rasio
kemungkinan, dan log-kemungkinan validasi silang."""
import warnings

import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from bab01_data import BENIH, bangkitkan, data_mini

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()


def icl(gm, X):
    R = gm.predict_proba(X)
    ent = -np.sum(R * np.log(np.clip(R, 1e-300, 1)))
    return gm.bic(X) + 2 * ent


# (1) Data mini
mu_g = np.array([X[g == k].mean(0) for k in (1, 2)])
print("(1) Data mini (m = 6):")
print("    K  jenis      p    log-kem    AIC      BIC")
for K, j in [(1, "full"), (2, "full"), (2, "tied"), (1, "spherical"),
             (2, "spherical")]:
    gm = GaussianMixture(K, covariance_type=j, reg_covar=1e-9, tol=1e-12,
                         max_iter=2000, means_init=None if K == 1 else mu_g,
                         random_state=BENIH).fit(X)
    print("    %d  %-9s %3d  %9.4f  %7.3f  %7.3f"
          % (K, j, gm._n_parameters(), gm.score(X) * 6, gm.aic(X), gm.bic(X)))

# (2) Uji rasio kemungkinan di bawah H0: K = 1
rng = np.random.default_rng(BENIH)
lrt = []
for _ in range(300):
    Y = rng.standard_normal((100, 2))
    l1 = GaussianMixture(1).fit(Y).score(Y) * 100
    l2 = GaussianMixture(2, n_init=5, random_state=0).fit(Y).score(Y) * 100
    lrt.append(2 * (l2 - l1))
lrt = np.array(lrt)
print("(2) 2(l_2 - l_1) di bawah H0: satu normal, m = 100;")
print("    300 ulangan")
print("    rata-rata = %.2f; chi2_6 rata-rata = 6" % lrt.mean())
print("    kuantil 95%%: simulasi = %.2f; chi2_6 = %.2f"
      % (np.quantile(lrt, 0.95), stats.chi2.ppf(0.95, 6)))
print("    P(2 dl > chi2_6(0.95)) = %.3f" % np.mean(lrt > stats.chi2.ppf(0.95, 6)))

# (3) Frekuensi K terpilih pada 100 data dari 4 komponen
mu_b = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], float)
S_b = np.array([np.eye(2) * 1.2] * 4)
pilih = {k: np.zeros(8, int) for k in ("AIC", "BIC", "ICL", "CV")}
for ul in range(100):
    Xb, _ = bangkitkan(400, np.full(4, 0.25), mu_b, S_b, rng)
    nilai = {k: [] for k in pilih}
    for K in range(1, 8):
        gm = GaussianMixture(K, n_init=3, random_state=ul).fit(Xb)
        nilai["AIC"].append(gm.aic(Xb))
        nilai["BIC"].append(gm.bic(Xb))
        nilai["ICL"].append(icl(gm, Xb))
        cv = 0.0
        for latih, uji in KFold(5, shuffle=True, random_state=ul).split(Xb):
            cv -= GaussianMixture(K, n_init=1, random_state=ul).fit(
                Xb[latih]).score(Xb[uji]) * len(uji)
        nilai["CV"].append(cv)
    for k in pilih:
        pilih[k][int(np.argmin(nilai[k])) + 1] += 1
print("(3) K terpilih pada 100 data (400 sampel, K benar = 4):")
print("    K       " + "".join("%5d" % K for K in range(1, 8)))
for k, v in pilih.items():
    print("    %-7s " % k + "".join("%5d" % c for c in v[1:]))
