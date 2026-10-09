"""Bab 11: klaster lunak, ARI, penaksiran kepadatan, skor anomali, dan
klasifikasi dengan GMM per kelas."""
import warnings

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from bab01_data import (BENIH, akhir_mini, bangkitkan, data_mini, log_normal,
                        logsumexp, log_gabung)

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()

# (1) ARI pada data mini
print("(1) ARI terhadap kelompok g:")
for nama, lab in [("EM dari (3,3), (5,5)", [1, 1, 1, 2, 2, 2]),
                  ("partisi {1,2,4} {3,5,6}", [1, 1, 2, 1, 2, 2]),
                  ("label ditukar", [2, 2, 2, 1, 1, 1])]:
    print("    %-25s ARI = %7.4f" % (nama, adjusted_rand_score(g, lab)))

# (2) Skor anomali: -log p(x) di bawah satu Gaussian dan GMM akhir
mu1 = X.mean(0)
S1 = np.cov(X.T, bias=True)
baru = np.array([[4.0, 4.0], [8.0, 0.0], [2.0, 2.0]])
lp_g = lambda Z: log_normal(Z, mu1, S1)
lp_m = lambda Z: logsumexp(log_gabung(Z, *akhir_mini()))
print("(2) -log p(x): satu Gaussian lawan GMM akhir")
print("    x        satu Gaussian   GMM")
for z in baru:
    print("    (%g, %g)   %8.4f      %8.4f"
          % (*z, -lp_g(z[None])[0], -lp_m(z[None])[0]))
print("    ambang (skor terbesar sampel latih): %.4f   %.4f"
      % (-lp_g(X).min(), -lp_m(X).min()))

# (3) Deteksi anomali pada simulasi: 600 normal + 30 anomali seragam
rng = np.random.default_rng(BENIH)
mu_b = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
S_b = np.array([[[1.0, 0.6], [0.6, 1.0]], [[1.0, -0.6], [-0.6, 1.0]],
                [[0.5, 0.0], [0.0, 1.5]]])
auc = {"satu Gaussian": [], "GMM K=3": []}
for ul in range(50):
    Xn, _ = bangkitkan(600, np.full(3, 1 / 3), mu_b, S_b, rng)
    Xa = rng.uniform([-4, -4], [10, 9], size=(30, 2))
    Z = np.vstack([Xn, Xa])
    y = np.r_[np.zeros(600), np.ones(30)]
    g1 = GaussianMixture(1).fit(Z)
    g3 = GaussianMixture(3, n_init=3, random_state=ul).fit(Z)
    auc["satu Gaussian"].append(roc_auc_score(y, -g1.score_samples(Z)))
    auc["GMM K=3"].append(roc_auc_score(y, -g3.score_samples(Z)))
print("(3) AUC skor -log p, 600 normal + 30 anomali, 50 ulangan:")
for k, v in auc.items():
    print("    %-14s %.3f" % (k, np.mean(v)))

# (4) Klasifikasi: setiap kelas campuran dua komponen yang berselang-seling
# sepanjang x1: kelas 0 di x1 = 0 dan 4, kelas 1 di x1 = 2 dan 6
def kelas(m, rng):
    s = np.diag([0.5, 1.0])
    X0, _ = bangkitkan(m, np.full(2, .5), np.array([[0., 0.], [4., 0.]]),
                       np.array([s, s]), rng)
    X1, _ = bangkitkan(m, np.full(2, .5), np.array([[2., 0.], [6., 0.]]),
                       np.array([s, s]), rng)
    return np.vstack([X0, X1]), np.r_[np.zeros(m), np.ones(m)]


Xl, yl = kelas(200, rng)
Xu, yu = kelas(5000, rng)
qda = QuadraticDiscriminantAnalysis().fit(Xl, yl)
lr = LogisticRegression().fit(Xl, yl)
gmm = [GaussianMixture(2, n_init=3, random_state=0).fit(Xl[yl == c])
       for c in (0, 1)]
skor = np.column_stack([gmm[c].score_samples(Xu) for c in (0, 1)])
print("(4) Akurasi data uji (selang-seling, 400 sampel latih):")
print("    regresi logistik      %.3f" % lr.score(Xu, yu))
print("    QDA (satu normal)     %.3f" % qda.score(Xu, yu))
print("    GMM K = 2 per kelas   %.3f" % np.mean(skor.argmax(1) == yu))
