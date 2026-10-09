"""Bab 2: sebaran normal multivariat. MLE rata-rata dan kovarians,
jarak Mahalanobis, bias pembagi m, dan sebaran chi-kuadrat d^2."""
import numpy as np
from scipy import stats

from bab01_data import BENIH, data_mini, log_normal

np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()
m, n = X.shape

# (1) MLE pada data mini, dicocokkan dengan NumPy dan SciPy
mu = X.mean(0)
S = (X - mu).T @ (X - mu) / m
val, vec = np.linalg.eigh(S)
Si = np.linalg.inv(S)
d2 = np.einsum("ij,jk,ik->i", X - mu, Si, X - mu)
mu_sp, S_sp = stats.multivariate_normal.fit(X)
print("(1) MLE satu Gaussian pada data mini:")
print("    mu =", mu, " Sigma =", S.tolist())
print("    np.cov(bias=True) =", np.cov(X.T, bias=True).tolist())
print("    SciPy fit: mu =", mu_sp, " Sigma =", S_sp.round(4).tolist())
print("    nilai eigen =", val, " det = %.4f" % np.linalg.det(S))
print("    d_i^2 =", d2, " jumlah = %.4f" % d2.sum())
print("    log-kemungkinan NumPy = %.4f  SciPy = %.4f"
      % (log_normal(X, mu, S).sum(),
         stats.multivariate_normal(mu, S).logpdf(X).sum()))

# (2) Bias pembagi m: rata-rata Sigma-hat atas banyak sampel kecil
rng = np.random.default_rng(BENIH)
S_benar = np.array([[2.0, -1.0], [-1.0, 2.0]])
L = np.linalg.cholesky(S_benar)
print("(2) Rata-rata Sigma-hat [s11 s12 s22], 20000 ulangan,")
print("    Sigma benar = [[2, -1], [-1, 2]]:")
for mm in (3, 6, 30):
    jumlah_m = np.zeros((2, 2))
    jumlah_m1 = np.zeros((2, 2))
    for _ in range(20000):
        Z = rng.standard_normal((mm, 2)) @ L.T
        D = Z - Z.mean(0)
        jumlah_m += D.T @ D / mm
        jumlah_m1 += D.T @ D / (mm - 1)
    a, b = jumlah_m / 20000, jumlah_m1 / 20000
    print("    m = %2d  pembagi m: [%.3f %.3f %.3f]  (m-1)/m = %.3f"
          % (mm, a[0, 0], a[0, 1], a[1, 1], (mm - 1) / mm))
    print("           pembagi m-1: [%.3f %.3f %.3f]"
          % (b[0, 0], b[0, 1], b[1, 1]))

# (3) d^2 dari parameter benar dan dari parameter taksiran
batas = stats.chi2.ppf(0.95, df=2)
Z = rng.standard_normal((100000, 2)) @ L.T
d2b = np.einsum("ij,jk,ik->i", Z, np.linalg.inv(S_benar), Z)
print("(3) Batas chi2_2(0.95) = %.4f" % batas)
print("    parameter benar: P(d^2 <= batas) = %.4f" % (d2b <= batas).mean())
print("                     rata-rata d^2    = %.4f" % d2b.mean())
for mm in (6, 30, 200):
    cakup = []
    for _ in range(5000):
        A = rng.standard_normal((mm, 2)) @ L.T
        mA = A.mean(0)
        SA = (A - mA).T @ (A - mA) / mm
        xb = rng.standard_normal(2) @ L.T        # sampel baru
        cakup.append((xb - mA) @ np.linalg.solve(SA, xb - mA) <= batas)
    print("    taksiran dari m = %3d: P(d^2 baru <= batas) = %.4f"
          % (mm, np.mean(cakup)))
