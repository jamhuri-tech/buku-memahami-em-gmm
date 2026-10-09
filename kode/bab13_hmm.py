"""Bab 13: hidden Markov model dengan emisi normal. Algoritma
maju-mundur dan Baum-Welch ditulis dari nol, diperiksa dengan
penjumlahan atas semua barisan keadaan."""
import itertools
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture

from bab01_data import BENIH, SIGMA0_2, awal_mini, data_mini, log_normal

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)


def emisi(X, mu, S):
    """B[t, k] = N(x_t | mu_k, Sigma_k)."""
    return np.exp(np.column_stack([log_normal(X, mu[k], S[k])
                                   for k in range(len(mu))]))


def maju_mundur(B, pi, A):
    """Alfa dan beta berskala; mengembalikan gamma, xi, log p(x)."""
    T, K = B.shape
    al = np.zeros((T, K))
    c = np.zeros(T)
    al[0] = pi * B[0]
    c[0] = al[0].sum()
    al[0] /= c[0]
    for t in range(1, T):
        al[t] = (al[t - 1] @ A) * B[t]
        c[t] = al[t].sum()
        al[t] /= c[t]
    be = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        be[t] = A @ (B[t + 1] * be[t + 1]) / c[t + 1]
    gam = al * be
    xi = np.array([al[t][:, None] * A * (B[t + 1] * be[t + 1])[None, :]
                   / c[t + 1] for t in range(T - 1)])
    return gam, xi, np.log(c).sum()


def baum_welch(X, pi, A, mu, S, iterasi=200):
    riw = []
    for _ in range(iterasi):
        gam, xi, lp = maju_mundur(emisi(X, mu, S), pi, A)
        riw.append(lp)
        pi = gam[0]
        A = xi.sum(0) / gam[:-1].sum(0)[:, None]
        N = gam.sum(0)
        mu = (gam.T @ X) / N[:, None]
        S = np.array([((gam[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / N[k]
                      + 1e-9 * np.eye(X.shape[1]) for k in range(len(N))])
    return pi, A, mu, S, riw


if __name__ == "__main__":
    X, g = data_mini()
    _, mu0, S0 = awal_mini()
    pi0 = np.array([0.5, 0.5])
    A0 = np.array([[0.75, 0.25], [0.25, 0.75]])

    # (1) Maju-mundur pada data mini sebagai barisan t = 1..6
    B = emisi(X, mu0, S0)
    gam, xi, lp = maju_mundur(B, pi0, A0)
    tot, post = 0.0, np.zeros(6)
    for z in itertools.product((0, 1), repeat=6):
        p = pi0[z[0]] * B[0, z[0]]
        for t in range(1, 6):
            p *= A0[z[t - 1], z[t]] * B[t, z[t]]
        tot += p
        post += p * (np.array(z) == 0)
    print("(1) Data mini sebagai barisan, A = [[3/4, 1/4], [1/4, 3/4]]:")
    print("    gamma_t(1) maju-mundur:")
    print("     ", gam[:, 0])
    print("    gamma_t(1) dari 64 barisan:")
    print("     ", post / tot)
    print("    log p(x) = %.4f  (64 barisan: %.4f)" % (lp, np.log(tot)))
    print("    rasio B[t,1]/B[t,2] =",
          " ".join("%.4g" % v for v in B[:, 0] / B[:, 1]))
    A1 = xi.sum(0) / gam[:-1].sum(0)[:, None]
    print("    A baru =", A1.round(4).tolist())

    # (2) Baum-Welch sampai konvergen pada data mini
    pi, A, mu, S, riw = baum_welch(X, pi0, A0, mu0, S0, iterasi=300)
    print("(2) Baum-Welch pada data mini:")
    print("    log p(x): %.4f -> %.4f" % (riw[0], riw[-1]))
    print("    A =", A.round(4).tolist())
    print("    mu =", mu.round(3).tolist())

    # (3) Simulasi: HMM lengket, emisi bertumpang tindih
    rng = np.random.default_rng(BENIH)
    A_b = np.array([[0.95, 0.05], [0.10, 0.90]])
    mu_b = np.array([[0.0, 0.0], [1.5, 1.0]])
    S_b = np.array([np.eye(2), np.eye(2)])
    T = 600
    z = np.zeros(T, int)
    for t in range(1, T):
        z[t] = rng.choice(2, p=A_b[z[t - 1]])
    Xs = np.array([rng.multivariate_normal(mu_b[k], S_b[k]) for k in z])
    gm = GaussianMixture(2, random_state=0).fit(Xs)
    lab_g = gm.predict(Xs)
    pi, A, mu, S, riw = baum_welch(Xs, np.array([.5, .5]),
                                   np.array([[.8, .2], [.2, .8]]),
                                   gm.means_, gm.covariances_, iterasi=200)
    gam, _, _ = maju_mundur(emisi(Xs, mu, S), pi, A)
    lab_h = gam.argmax(1)
    acc = lambda lab: max(np.mean(lab == z), np.mean(lab != z))
    urut = np.argsort(mu[:, 0])
    print("(3) T = 600, A benar = [[0.95, 0.05], [0.10, 0.90]]:")
    print("    akurasi keadaan: GMM %.3f, HMM %.3f" % (acc(lab_g), acc(lab_h)))
    print("    A taksiran =", A[np.ix_(urut, urut)].round(3).tolist())
