"""Bab 12: variasi EM. EM keras, EM daring, campuran t, dan Bayes
variasional dengan BayesianGaussianMixture."""
import warnings

import numpy as np
from scipy.special import gammaln
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from bab01_data import (BENIH, awal_mini, bangkitkan, data_mini, em,
                        langkah_m, log_gabung, logsumexp, tanggung_jawab)

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()


def em_keras(X, pi, mu, S, iterasi=100):
    for _ in range(iterasi):
        A = log_gabung(X, pi, mu, S)
        R = np.eye(len(pi))[A.argmax(1)]
        pi, mu, S = langkah_m(X, R)
    return pi, mu, S


def em_t(X, mu, S, pi, nu=4.0, iterasi=200):
    m, n = X.shape
    K = len(pi)
    for _ in range(iterasi):
        A = np.empty((m, K))
        U = np.empty((m, K))
        for k in range(K):
            D = X - mu[k]
            d2 = np.einsum("ij,jk,ik->i", D, np.linalg.inv(S[k]), D)
            A[:, k] = (np.log(pi[k]) + gammaln((nu + n) / 2)
                       - gammaln(nu / 2) - n / 2 * np.log(nu * np.pi)
                       - 0.5 * np.log(np.linalg.det(S[k]))
                       - (nu + n) / 2 * np.log1p(d2 / nu))
            U[:, k] = (nu + n) / (nu + d2)
        R = np.exp(A - logsumexp(A)[:, None])
        N = R.sum(0)
        pi = N / m
        W = R * U
        mu = (W.T @ X) / W.sum(0)[:, None]
        S = np.array([((W[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / N[k]
                      for k in range(K)])
    return mu, S, pi, (R * U).sum(1)



if __name__ == "__main__":
    # (1) EM keras pada data mini
    pi, mu, S = em_keras(X, *awal_mini(), iterasi=1)
    print("(1) EM keras, satu iterasi dari awal Bab 1:")
    print("    mu =", mu.tolist())
    print("    Sigma_1 =", S[0].tolist())

    # (2) Bias EM keras pada komponen yang bertumpang tindih
    rng = np.random.default_rng(BENIH)
    mu_b = np.array([[0.0, 0.0], [2.0, 0.0]])
    S_b = np.array([np.eye(2)] * 2)
    Xb, _ = bangkitkan(4000, np.array([.5, .5]), mu_b, S_b, rng)
    awal = (np.array([.5, .5]), mu_b.copy(), S_b.copy())
    _, mu_l, S_l, _ = em(Xb, *awal, iterasi=500, tol=1e-9)
    _, mu_k, S_k = em_keras(Xb, *awal, iterasi=200)
    print("(2) Dua komponen N((0,0), I) dan N((2,0), I), m = 4000:")
    print("    EM lunak: mu_1 =", mu_l[0].round(3), " var x1 = %.3f" % S_l[0][0, 0])
    print("    EM keras: mu_1 =", mu_k[0].round(3), " var x1 = %.3f" % S_k[0][0, 0])

    # (3) EM daring: rata-rata statistik cukup dengan langkah gamma_t
    pi_b = np.array([0.5, 0.3, 0.2])
    mu3 = np.array([[0.0, 0.0], [5.0, 1.0], [2.0, 5.0]])
    S3 = np.array([[[2.0, 0.8], [0.8, 1.0]], [[1.0, -0.6], [-0.6, 1.5]],
                   [[0.5, 0.0], [0.0, 2.0]]])
    Xs, _ = bangkitkan(20000, pi_b, mu3, S3, rng)
    Xu, _ = bangkitkan(5000, pi_b, mu3, S3, rng)
    pi, mu, S = np.full(3, 1 / 3), Xs[:3].copy(), np.array([np.eye(2) * 4] * 3)
    s0, s1, s2 = pi.copy(), pi[:, None] * mu, \
        np.array([pi[k] * (S[k] + np.outer(mu[k], mu[k])) for k in range(3)])
    print("(3) EM daring, satu sampel per langkah,")
    print("    gamma_t = (t + 10)^-0.6:")
    for t, x in enumerate(Xs, start=1):
        a = log_gabung(x[None], pi, mu, S)[0]
        r = np.exp(a - logsumexp(a[None])[0])
        gam = (t + 10) ** -0.6
        s0 = (1 - gam) * s0 + gam * r
        s1 = (1 - gam) * s1 + gam * r[:, None] * x
        s2 = (1 - gam) * s2 + gam * r[:, None, None] * np.outer(x, x)
        if t > 20:
            pi = s0 / s0.sum()
            mu = s1 / s0[:, None]
            S = s2 / s0[:, None, None] - np.einsum("ki,kj->kij", mu, mu)
            S += 1e-6 * np.eye(2)
        if t in (1000, 5000, 20000):
            print("    t = %5d  log-kem. uji per sampel = %.4f"
                  % (t, logsumexp(log_gabung(Xu, pi, mu, S)).mean()))
    gb = GaussianMixture(3, n_init=3, random_state=0).fit(Xs)
    print("    EM biasa (20000 sampel):            %.4f" % gb.score(Xu))

    # (4) Campuran t: kokoh terhadap pencilan
    Xo, _ = bangkitkan(300, np.array([.5, .5]), np.array([[0., 0.], [5., 0.]]),
                       np.array([np.eye(2)] * 2), rng)
    Xo = np.vstack([Xo, rng.uniform([-30, -30], [30, 30], size=(15, 2))])


    awal_mu = np.array([[0.5, 0.5], [4.5, -0.5]])
    mu_t, S_t, pi_t, U = em_t(Xo, awal_mu, np.array([np.eye(2) * 3] * 2),
                              np.array([.5, .5]))
    gmo = GaussianMixture(2, means_init=awal_mu, random_state=0).fit(Xo)
    print("(4) 300 sampel + 15 pencilan; pusat benar (0, 0) dan (5, 0):")
    print("    GMM:        mu =", gmo.means_.round(2).tolist())
    print("                var x1 =", gmo.covariances_[:, 0, 0].round(2))
    print("    campuran t: mu =", mu_t.round(2).tolist())
    print("                var x1 =", S_t[:, 0, 0].round(2))
    print("    bobot u rata-rata: inlier %.3f, pencilan %.3f"
          % (U[:300].mean(), U[300:].mean()))

    # (5) Bayes variasional: K = 10 pada data 3 komponen
    bg = BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.01,
                                 max_iter=2000, random_state=0).fit(Xs[:2000])
    print("(5) BayesianGaussianMixture, K = 10, prior konsentrasi 0.01:")
    w = np.sort(bg.weights_)[::-1]
    print("    bobot terurut:", " ".join("%.3f" % v for v in w[:5]))
    print("                  ", " ".join("%.3f" % v for v in w[5:]))
    print("    komponen dengan bobot > 0.01: %d" % (bg.weights_ > 0.01).sum())
