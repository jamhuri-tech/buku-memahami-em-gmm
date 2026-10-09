"""Data mini yang dipakai di seluruh buku, dan fungsi dasar GMM.

Enam pelanggan sebuah toko, dua fitur: x1 = frekuensi belanja per
bulan, x2 = nilai belanja per kunjungan (ratusan ribu rupiah). Kolom g
adalah kelompok pembanding dari catatan toko (1 = biasa, 2 = setia);
EM tidak memakainya. Kelompok 2 adalah cerminan kelompok 1 terhadap
titik (4, 4).
"""
import numpy as np

BENIH = 20261009

DATA_MINI = np.array([[1, 1, 1],
                      [1, 4, 1],
                      [4, 1, 1],
                      [4, 7, 2],
                      [7, 4, 2],
                      [7, 7, 2]], dtype=float)

# Parameter awal EM yang dipakai sejak Bab 1: dua komponen berbobot
# sama, rata-rata (3, 3) dan (5, 5), kovarians sigma0^2 I dengan
# sigma0^2 = 6 / log 3, sehingga N(x | mu, sigma0^2 I) sebanding dengan
# 3^(-||x - mu||^2 / 12).
SIGMA0_2 = 6 / np.log(3)


def data_mini():
    """Mengembalikan (X, g): X berukuran 6 x 2, g kelompok pembanding."""
    return DATA_MINI[:, :2].copy(), DATA_MINI[:, 2].astype(int)


def awal_mini():
    """Parameter awal (pi, mu, Sigma) untuk K = 2."""
    pi = np.array([0.5, 0.5])
    mu = np.array([[3.0, 3.0], [5.0, 5.0]])
    Sigma = np.array([SIGMA0_2 * np.eye(2)] * 2)
    return pi, mu, Sigma


def log_normal(X, mu, Sigma):
    """log N(x_i | mu, Sigma) untuk setiap baris X, lewat Cholesky.

    Sigma = L L^T, sehingga log det Sigma = 2 sum log L_jj dan
    (x - mu)^T Sigma^-1 (x - mu) = ||L^-1 (x - mu)||^2.
    """
    X = np.atleast_2d(X)
    n = X.shape[1]
    L = np.linalg.cholesky(Sigma)
    u = np.linalg.solve(L, (X - mu).T)          # n x m
    d2 = np.sum(u**2, axis=0)                    # jarak Mahalanobis^2
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (n * np.log(2 * np.pi) + logdet + d2)


def logsumexp(A, axis=1):
    """log sum exp tanpa luapan: geser dengan maksimum lebih dahulu."""
    a = np.max(A, axis=axis, keepdims=True)
    return np.squeeze(a, axis=axis) + np.log(
        np.sum(np.exp(A - a), axis=axis))


def log_gabung(X, pi, mu, Sigma):
    """Matriks a_ik = log pi_k + log N(x_i | mu_k, Sigma_k), m x K."""
    return np.column_stack([np.log(pi[k]) + log_normal(X, mu[k], Sigma[k])
                            for k in range(len(pi))])


def tanggung_jawab(X, pi, mu, Sigma):
    """Mengembalikan (R, ll): R_ik = r_ik dan ll = log-likelihood total."""
    A = log_gabung(X, pi, mu, Sigma)
    lp = logsumexp(A)                            # log p(x_i)
    return np.exp(A - lp[:, None]), float(lp.sum())


def langkah_m(X, R):
    """Pembaruan bentuk tertutup (Bab 4): pi, mu, Sigma dari R."""
    m = X.shape[0]
    N = R.sum(axis=0)
    pi = N / m
    mu = (R.T @ X) / N[:, None]
    Sigma = np.array([((R[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / N[k]
                      for k in range(R.shape[1])])
    return pi, mu, Sigma


def em(X, pi, mu, Sigma, iterasi=200, tol=1e-10):
    """EM untuk GMM kovarians penuh. Mengembalikan parameter dan riwayat
    log-likelihood (nilai sebelum setiap langkah M, lalu nilai akhir)."""
    riwayat = []
    for _ in range(iterasi):
        R, ll = tanggung_jawab(X, pi, mu, Sigma)
        riwayat.append(ll)
        if len(riwayat) > 1 and abs(riwayat[-1] - riwayat[-2]) < tol:
            break
        pi, mu, Sigma = langkah_m(X, R)
    return pi, mu, Sigma, riwayat


def akhir_mini():
    """Parameter GMM data mini sesudah EM konvergen, dibulatkan ke nilai
    kelompok: pi = (1/2, 1/2), mu = (2, 2) dan (6, 6),
    Sigma = [[2, -1], [-1, 2]] untuk kedua komponen."""
    pi = np.array([0.5, 0.5])
    mu = np.array([[2.0, 2.0], [6.0, 6.0]])
    S = np.array([[2.0, -1.0], [-1.0, 2.0]])
    return pi, mu, np.array([S, S])


def bangkitkan(m, pi, mu, Sigma, rng):
    """Resep generatif: z_i ~ Kategorik(pi), x_i ~ N(mu_z, Sigma_z).
    Mengembalikan (X, z) dengan z bernilai 0, ..., K-1."""
    z = rng.choice(len(pi), size=m, p=pi)
    X = np.empty((m, mu.shape[1]))
    for k in range(len(pi)):
        s = z == k
        L = np.linalg.cholesky(Sigma[k])
        X[s] = mu[k] + rng.standard_normal((s.sum(), mu.shape[1])) @ L.T
    return X, z
