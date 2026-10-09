"""GMM kovarians penuh dengan NumPy saja: kepadatan, langkah E,
langkah M, EM, BIC, dan skor anomali."""
import numpy as np


def log_normal(X, mu, S):
    L = np.linalg.cholesky(S)              # S = L L^T
    U = np.linalg.solve(L, (X - mu).T)     # L^-1 (x - mu)
    n = X.shape[1]
    logdet = 2 * np.log(np.diag(L)).sum()
    return -0.5 * (n * np.log(2 * np.pi) + logdet
                   + (U ** 2).sum(0))


def langkah_e(X, pi, mu, S):
    A = np.column_stack([np.log(pi[k])
                         + log_normal(X, mu[k], S[k])
                         for k in range(len(pi))])
    a = A.max(1, keepdims=True)            # log-sum-exp
    lp = a[:, 0] + np.log(np.exp(A - a).sum(1))
    return np.exp(A - lp[:, None]), lp.sum()


def langkah_m(X, R, eps=1e-6):
    N = R.sum(0)
    mu = R.T @ X / N[:, None]
    S = []
    for k in range(len(N)):
        D = X - mu[k]
        S.append((R[:, k, None] * D).T @ D / N[k]
                 + eps * np.eye(X.shape[1]))
    return N / len(X), mu, np.array(S)


def em(X, pi, mu, S, tol=1e-8, iterasi=500):
    lama = -np.inf
    for t in range(iterasi):
        R, ll = langkah_e(X, pi, mu, S)
        if ll - lama < tol:
            break
        lama = ll
        pi, mu, S = langkah_m(X, R)
    return pi, mu, S, R, ll, t


def bic(ll, K, n, m):
    p = (K - 1) + K * n + K * n * (n + 1) // 2
    return -2 * ll + p * np.log(m)


if __name__ == "__main__":
    X = np.array([[1, 1], [1, 4], [4, 1],
                  [4, 7], [7, 4], [7, 7]], float)
    pi0 = np.array([0.5, 0.5])
    mu0 = np.array([[3.0, 3.0], [5.0, 5.0]])
    S0 = np.array([6 / np.log(3) * np.eye(2)] * 2)
    R, ll0 = langkah_e(X, pi0, mu0, S0)
    print("r_i1 awal:", R[:, 0].round(4), " l = %.4f" % ll0)
    pi, mu, S, R, ll, t = em(X, pi0, mu0, S0)
    print("sesudah %d iterasi: l = %.4f" % (t, ll))
    print("pi =", pi.round(4), " mu =", mu.round(3).tolist())
    print("Sigma_1 =", S[0].round(3).tolist())
    print("BIC K = 2: %.3f" % bic(ll, 2, 2, 6))
    z = np.array([[4.0, 4.0]])
    print("skor anomali (4, 4): %.4f"
          % -langkah_e(z, pi, mu, S)[1])
