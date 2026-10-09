"""Bab 3: model generatif dengan variabel laten, posterior Bayes,
hukum varians total, dan kemungkinan yang tak terbatas."""
import numpy as np
from sklearn.mixture import GaussianMixture

from bab01_data import (BENIH, akhir_mini, bangkitkan, data_mini,
                        log_normal, tanggung_jawab)

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(BENIH)
X, g = data_mini()
pi, mu, S = akhir_mini()

# (1) Resep generatif dan hukum varians total
Xb, z = bangkitkan(100000, pi, mu, S, rng)
W = sum(pi[k] * S[k] for k in range(2))
mbar = pi @ mu
B = sum(pi[k] * np.outer(mu[k] - mbar, mu[k] - mbar) for k in range(2))
print("(1) 100000 sampel dari GMM akhir data mini:")
print("    bagian z = 1:", round((z == 0).mean(), 4))
print("    rata-rata sampel =", Xb.mean(0), " teori =", mbar)
print("    kovarians sampel =", np.cov(Xb.T, bias=True).round(3).tolist())
print("    W + B            =", (W + B).tolist())

# (2) Posterior dan batas keputusan
Si = np.linalg.inv(S[0])
w = Si @ (mu[0] - mu[1])
w0 = -0.5 * (mu[0] @ Si @ mu[0] - mu[1] @ Si @ mu[1]) + np.log(pi[0] / pi[1])
baru = np.array([[4, 1], [4, 4], [5, 4]], float)
R, _ = tanggung_jawab(baru, pi, mu, S)
gm = GaussianMixture(2, covariance_type="tied")
gm.weights_, gm.means_, gm.covariances_ = pi, mu, S[0]
gm.precisions_cholesky_ = np.linalg.cholesky(Si)
print("(2) log r1/r2 = w'x + w0 dengan w =", w, " w0 = %.1f" % w0)
print("    x        w'x + w0   r1 NumPy   r1 sklearn")
for x, r, rs in zip(baru, R[:, 0], gm.predict_proba(baru)[:, 0]):
    print("    (%g, %g)   %6.1f   %.6f   %.6f" % (*x, w @ x + w0, r, rs))

# (3) Kemungkinan tak terbatas: mu_1 = x_1, Sigma_1 = eps I
mu_s = X.mean(0)
S_s = (X - mu_s).T @ (X - mu_s) / 6
print("(3) mu_1 = x_1, Sigma_1 = eps I; komponen 2 = satu Gaussian")
for eps in (1.0, 0.1, 0.01, 1e-4, 1e-8):
    _, ll = tanggung_jawab(X, pi, np.array([X[0], mu_s]),
                           np.array([eps * np.eye(2), S_s]))
    print("    eps = %-7g log-kemungkinan = %9.4f" % (eps, ll))
print("    pembanding: EM data mini     = %9.4f"
      % tanggung_jawab(X, *akhir_mini())[1])
