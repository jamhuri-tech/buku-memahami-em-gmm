"""Bab 10: komponen runtuh, reg_covar, dan EM-MAP dengan prior
Dirichlet dan Wishart terbalik."""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from bab01_data import BENIH, bangkitkan, data_mini, logsumexp, log_gabung

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()
X7 = np.vstack([X, [1.0, 1.0]])               # sampel 1 tercatat dua kali


def em_map(X, pi, mu, S, alpha=1.0, Psi=None, nu=None, iterasi=200,
           cetak=(), eps=0.0):
    """EM-MAP: prior Dirichlet(alpha) pada pi dan Wishart terbalik
    (Psi, nu) pada setiap Sigma_k. alpha = 1 dan Psi = None: EM biasa."""
    m, n = X.shape
    K = len(pi)
    for t in range(iterasi):
        A = log_gabung(X, pi, mu, S)
        lp = logsumexp(A)
        R = np.exp(A - lp[:, None])
        if t in cetak:
            print("    t = %d  l = %9.4f  det Sigma_1 = %.4f"
                  % (t, lp.sum(), np.linalg.det(S[0])))
        N = R.sum(0)
        pi = (N + alpha - 1) / (m + K * (alpha - 1))
        mu = (R.T @ X) / N[:, None]
        S = []
        for k in range(K):
            D = X - mu[k]
            Sk = (R[:, k, None] * D).T @ D
            if Psi is None:
                S.append(Sk / N[k] + eps * np.eye(n))
            else:
                S.append((Sk + Psi) / (N[k] + nu + n + 1))
        S = np.array(S)
    return pi, mu, S, lp.sum()


pi0 = np.full(3, 1 / 3)
mu0 = np.array([[1.0, 1.0], [2.0, 3.0], [6.0, 6.0]])
S0 = np.array([np.eye(2) * 2.0] * 3)


if __name__ == "__main__":
    # (1) Runtuh: K = 3, satu komponen dimulai di titik ganda (1, 1)
    print("(1) Data mini + (1, 1) ganda, K = 3, EM tanpa regularisasi:")
    with np.errstate(all="ignore"):
        for t in range(1, 40):
            try:
                em_map(X7, pi0, mu0, S0, iterasi=t, cetak=(t - 1,) if t < 4 else ())
            except np.linalg.LinAlgError:
                print("    iterasi %d: Sigma_1 tidak lagi definit positif" % (t - 1))
                break
    try:
        GaussianMixture(3, reg_covar=0.0, means_init=mu0).fit(X7)
    except ValueError:
        print("    sklearn reg_covar = 0: ValueError")
    print("    EM dengan Sigma_k + eps I:")
    for eps in (1e-6, 1e-2, 0.5):
        pi, mu, S, lp = em_map(X7, pi0, mu0, S0, iterasi=300, eps=eps)
        print("    eps = %-6g l = %8.3f  Sigma_1 = %s"
              % (eps, lp, np.diag(S[0]).round(4)))

    # (2) EM-MAP dengan Dirichlet(2) dan Wishart terbalik (Psi = 0.5 I, nu = 1)
    pi, mu, S, lp = em_map(X7, pi0, mu0, S0, alpha=2.0, Psi=0.5 * np.eye(2),
                           nu=1.0, iterasi=500)
    print("(2) EM-MAP, alpha = 2, Psi = 0.5 I, nu = 1:")
    print("    pi =", pi)
    for k in range(3):
        print("    mu_%d = %s" % (k + 1, mu[k].round(2)))
        print("      Sigma_%d = %s" % (k + 1, S[k].round(3).tolist()))

    # (3) Data kecil berdimensi tinggi: log-kemungkinan data uji
    rng = np.random.default_rng(BENIH)
    n = 6
    mu_b = np.array([np.zeros(n), np.r_[3, np.zeros(n - 1)],
                     np.r_[0, 3, np.zeros(n - 2)]])
    A = rng.standard_normal((n, n)) * 0.3
    S_b = np.array([np.eye(n) + A @ A.T] * 3)
    hasil = {rc: [] for rc in (1e-6, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0)}
    for ul in range(50):
        Xl, _ = bangkitkan(45, np.full(3, 1 / 3), mu_b, S_b, rng)
        Xu, _ = bangkitkan(2000, np.full(3, 1 / 3), mu_b, S_b, rng)
        for rc in hasil:
            gm = GaussianMixture(3, reg_covar=rc, n_init=3,
                                 random_state=ul).fit(Xl)
            hasil[rc].append(gm.score(Xu))
    print("(3) m = 45 sampel latih, n = 6, K = 3; 50 ulangan;")
    print("    rata-rata log-kemungkinan per sampel uji:")
    for rc, v in hasil.items():
        print("    reg_covar = %-6g %8.3f" % (rc, np.mean(v)))
