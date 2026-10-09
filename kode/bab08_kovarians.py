"""Bab 8: struktur kovarians GMM dan K-Means sebagai limitnya."""
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from bab01_data import BENIH, bangkitkan, data_mini

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()
JENIS = ("full", "tied", "diag", "spherical")

# (1) Empat struktur pada data mini, pemulaan dari kelompok g
R0 = np.column_stack([g == 1, g == 2]).astype(float)
mu0 = np.array([X[g == k].mean(0) for k in (1, 2)])
print("(1) Data mini, EM dari kelompok g:")
print("    %-10s %-6s %-10s %s" % ("jenis", "param", "log-kem", "kovarians komp. 1"))
for j in JENIS:
    gm = GaussianMixture(2, covariance_type=j, means_init=mu0, tol=1e-12,
                         max_iter=2000, reg_covar=1e-9,
                         random_state=BENIH).fit(X)
    c = gm.covariances_
    if j == "full":
        c1 = c[0]
    elif j == "tied":
        c1 = c
    elif j == "diag":
        c1 = np.diag(c[0])
    else:
        c1 = c[0] * np.eye(2)
    print("    %-10s %-6d %-10.4f %s" % (j, gm._n_parameters(),
                                       gm.score(X) * 6, c1.round(3).tolist()))

# (2) K-Means sebagai limit: tanggung jawab sampel 3 untuk sigma^2 mengecil
print("(2) r_31; mu = (3,3), (5,5), Sigma = s2 I, pi = (1/2, 1/2):")
for s2 in (6 / np.log(3), 4.0, 1.0, 0.25, 0.05):
    print("    s2 = %-7.4f r_31 = %.6f" % (s2, 1 / (1 + np.exp(-12 / (2 * s2)))))
km = KMeans(2, init=np.array([[3., 3.], [5., 5.]]), n_init=1).fit(X)
print("    KMeans: label =", km.labels_, " inersia = %.1f" % km.inertia_)
print("            pusat =", km.cluster_centers_.tolist())

# (3) Klaster memanjang: empat struktur dan K-Means
rng = np.random.default_rng(BENIH)
S_a = np.array([[4.0, 3.6], [3.6, 4.0]])
S_b = np.array([[4.0, -3.6], [-3.6, 4.0]])
Xs, zs = bangkitkan(600, np.array([0.5, 0.5]), np.array([[0., 0.], [3., 0.]]),
                    np.array([S_a, S_b]), rng)
print("(3) 600 sampel, dua klaster memanjang bersilang:")
print("    %-10s %-6s %-12s %s" % ("jenis", "param", "BIC", "ARI"))
for j in JENIS:
    gm = GaussianMixture(2, covariance_type=j, n_init=10,
                         random_state=BENIH).fit(Xs)
    print("    %-10s %-6d %-12.1f %.3f" % (j, gm._n_parameters(), gm.bic(Xs),
                                         adjusted_rand_score(zs, gm.predict(Xs))))
km = KMeans(2, n_init=10, random_state=BENIH).fit(Xs)
print("    %-10s %-6s %-12s %.3f" % ("KMeans", "-", "-",
                                    adjusted_rand_score(zs, km.labels_)))
