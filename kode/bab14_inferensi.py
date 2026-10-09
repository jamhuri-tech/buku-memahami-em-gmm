"""Bab 14: galat baku taksiran GMM dari matriks informasi (hasil kali
luar skor), bootstrap parametrik, dan cakupan selang Wald."""
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture

from bab01_data import (BENIH, awal_mini, bangkitkan, data_mini, log_gabung,
                        logsumexp)

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)


def skor(X, pi, mu, S):
    """Skor per sampel untuk (eta, mu_k, vech Sigma_k), K = 2.
    eta = logit pi_1, sehingga d l_i / d eta = r_i1 - pi_1."""
    A = log_gabung(X, pi, mu, S)
    R = np.exp(A - logsumexp(A)[:, None])
    m, n = X.shape
    kol = [R[:, 0] - pi[0]]
    for k in range(2):
        Si = np.linalg.inv(S[k])
        D = X - mu[k]
        U = D @ Si                                   # baris: Sigma^-1 (x - mu)
        for j in range(n):
            kol.append(R[:, k] * U[:, j])
        for a in range(n):
            for b in range(a, n):
                g = 0.5 * R[:, k] * (U[:, a] * U[:, b] - Si[a, b])
                kol.append(g if a == b else 2 * g)
    return np.column_stack(kol)


def galat_baku(X, pi, mu, S):
    G = skor(X, pi, mu, S)
    return np.sqrt(np.diag(np.linalg.inv(G.T @ G)))


def urutkan(gm):
    """Menyelaraskan label: komponen 1 adalah yang x1-nya lebih kecil."""
    o = np.argsort(gm.means_[:, 0])
    return gm.weights_[o], gm.means_[o], gm.covariances_[o]


if __name__ == "__main__":
    # (1) Model pi saja: informasi teramati dan lengkap
    X, g = data_mini()
    _, mu0, S0 = awal_mini()
    rho = np.exp(log_gabung(X, np.array([.5, .5]), mu0, S0) @ [1, -1])
    l = lambda p: np.log(p * rho + 1 - p).sum()
    h = 1e-4
    Iobs = -(l(0.5 + h) - 2 * l(0.5) + l(0.5 - h)) / h ** 2
    print("(1) Model pi saja di pi = 1/2:")
    print("    I_obs (beda hingga) = %.4f  SE = %.4f" % (Iobs, Iobs ** -0.5))
    print("    I_c = 24.0000              SE = %.4f" % 24 ** -0.5)

    # (2) Galat baku pada data simulasi m = 200
    rng = np.random.default_rng(BENIH)
    pi_b = np.array([0.3, 0.7])
    mu_b = np.array([[0.0, 0.0], [3.0, 1.0]])
    S_b = np.array([[[1.0, 0.3], [0.3, 1.0]], [[1.5, -0.4], [-0.4, 1.0]]])
    Xs, _ = bangkitkan(200, pi_b, mu_b, S_b, rng)
    gm = GaussianMixture(2, n_init=5, random_state=0).fit(Xs)
    pi, mu, S = urutkan(gm)
    se = galat_baku(Xs, pi, mu, S)
    se_pi = se[0] * pi[0] * pi[1]                    # metode delta dari eta
    boot = []
    for b in range(200):
        Xb, _ = bangkitkan(200, pi, mu, S, rng)
        p_, m_, _ = urutkan(GaussianMixture(2, n_init=2, random_state=b)
                            .fit(Xb))
        boot.append(np.r_[p_[0], m_.ravel()])
    boot = np.array(boot)
    mc = []
    for r in range(300):
        Xr, _ = bangkitkan(200, pi_b, mu_b, S_b, rng)
        p_, m_, _ = urutkan(GaussianMixture(2, n_init=2, random_state=r).fit(Xr))
        mc.append(np.r_[p_[0], m_.ravel()])
    mc = np.array(mc)
    nama = ["pi_1", "mu_11", "mu_12", "mu_21", "mu_22"]
    se_skor = np.r_[se_pi, se[1:3], se[6:8]]
    print("(2) m = 200; galat baku menurut tiga cara:")
    print("    SE(eta) skor = %.3f, eta = logit pi_1 = %.3f"
          % (se[0], np.log(pi[0] / pi[1])))
    print("    param   taksiran  skor    bootstrap  Monte Carlo")
    est = np.r_[pi[0], mu.ravel()]
    for j, nm in enumerate(nama):
        print("    %-6s  %7.3f  %6.3f   %6.3f     %6.3f"
              % (nm, est[j], se_skor[j], boot[:, j].std(), mc[:, j].std()))

    # (3) Cakupan selang Wald 95% untuk mu_21 = 3
    hit, hit_pi = 0, 0
    for r in range(300):
        Xr, _ = bangkitkan(200, pi_b, mu_b, S_b, rng)
        p_, m_, s_ = urutkan(GaussianMixture(2, n_init=2, random_state=r).fit(Xr))
        e = galat_baku(Xr, p_, m_, s_)
        hit += abs(m_[1, 0] - 3.0) <= 1.96 * e[6]
        eta, se_eta = np.log(p_[0] / p_[1]), e[0]
        lo, hi = eta - 1.96 * se_eta, eta + 1.96 * se_eta
        hit_pi += lo <= np.log(0.3 / 0.7) <= hi
    print("(3) Cakupan selang 95%, 300 ulangan:")
    print("    mu_21 (Wald):          %.3f" % (hit / 300))
    print("    pi_1 (Wald pada logit): %.3f" % (hit_pi / 300))
