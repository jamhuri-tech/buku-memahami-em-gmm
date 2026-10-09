"""Bab 1: data mini, kepadatan campuran, tanggung jawab, dan turunan
log-likelihood satu sampel, dicocokkan dengan SciPy dan scikit-learn."""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from bab01_data import (BENIH, awal_mini, data_mini, log_normal,
                        tanggung_jawab)

np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()
m, n = X.shape

# (1) Satu Gaussian lawan dua komponen
mu1 = X.mean(axis=0)
S1 = (X - mu1).T @ (X - mu1) / m
ll1 = log_normal(X, mu1, S1).sum()
gm = GaussianMixture(2, covariance_type="full", tol=1e-12,
                     max_iter=1000, reg_covar=0.0,
                     means_init=[[3, 3], [5, 5]],
                     random_state=BENIH).fit(X)
print("(1) Data mini: m = %d sampel, n = %d fitur" % (m, n))
print("    satu Gaussian: mu =", mu1, " det Sigma = %.4f" % np.linalg.det(S1))
print("    log-kemungkinan satu Gaussian = %.4f" % ll1)
print("    GaussianMixture K = 2:")
print("      pi =", gm.weights_, " mu =", gm.means_.round(4).tolist())
print("      log-kemungkinan            = %.4f" % (gm.score(X) * m))

# (2) Tanggung jawab dengan parameter awal
pi, mu, Sigma = awal_mini()
R, ll = tanggung_jawab(X, pi, mu, Sigma)
p_scipy = np.column_stack([pi[k] * multivariate_normal(mu[k], Sigma[k]).pdf(X)
                           for k in range(2)])
ref = GaussianMixture(2, covariance_type="full")
ref.weights_, ref.means_, ref.covariances_ = pi, mu, Sigma
ref.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(S))
                                     for S in Sigma])
print("(2) Awal: mu1 = (3, 3), mu2 = (5, 5), sigma0^2 = %.4f"
      % Sigma[0, 0, 0])
print("    r_i1 NumPy   :", R[:, 0])
print("    r_i1 SciPy   :", (p_scipy[:, 0] / p_scipy.sum(1)))
print("    r_i1 sklearn :", ref.predict_proba(X)[:, 0])
print("    log-kemungkinan awal = %.4f" % ll)

# (3) Turunan log p(x_1) terhadap mu_1, Sigma_1, dan beta_1
i = 0
x = X[i]
r = R[i, 0]
Si = np.linalg.inv(Sigma[0])
d = x - mu[0]
g_mu = r * Si @ d
g_S = 0.5 * r * (Si @ np.outer(d, d) @ Si - Si)
g_beta = r - pi[0]


def lp(pi_, mu_, S_):
    return tanggung_jawab(X[i:i + 1], pi_, mu_, S_)[1]


h = 1e-6
fd_mu = []
for j in range(n):
    e = np.zeros((2, n))
    e[0, j] = h
    fd_mu.append((lp(pi, mu + e, Sigma) - lp(pi, mu - e, Sigma)) / (2 * h))
E = np.zeros_like(Sigma)
E[0, 0, 0] = h
fd_S11 = (lp(pi, mu, Sigma + E) - lp(pi, mu, Sigma - E)) / (2 * h)
b = np.log(pi)
fd_b = []
for s in (h, -h):
    bb = b + np.array([s, 0])
    fd_b.append(lp(np.exp(bb) / np.exp(bb).sum(), mu, Sigma))
print("(3) Turunan log p(x_1), x_1 = (1, 1):")
fmt = lambda v: np.array2string(np.atleast_1d(v), precision=4,
                                suppress_small=True)
print("    %-11s %-19s %s" % ("", "rumus", "beda hingga"))
for nama, a, b in [("d/d mu_1", g_mu, np.array(fd_mu)),
                   ("d/d S1_11", g_S[0, 0], fd_S11),
                   ("d/d beta_1", g_beta, (fd_b[0] - fd_b[1]) / (2 * h))]:
    print("    %-11s %-19s %s" % (nama, fmt(a), fmt(b)))
