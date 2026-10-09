"""Model Bab 6: data survei biner dan EM campuran Bernoulli."""
import numpy as np

# Campuran Bernoulli: 6 responden x 3 butir ya/tidak
# (belanja daring, memakai promo, ikut program poin)
B = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0],
              [1, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=float)


def bern_e(B, pi, th):
    th = np.clip(th, 1e-300, 1 - 1e-16)
    A = np.log(pi) + B @ np.log(th).T + (1 - B) @ np.log(1 - th).T
    a = A.max(1, keepdims=True)
    lp = a[:, 0] + np.log(np.exp(A - a).sum(1))
    return np.exp(A - lp[:, None]), lp.sum()


def bern_m(B, R):
    N = R.sum(0)
    return N / len(B), (R.T @ B) / N[:, None]
