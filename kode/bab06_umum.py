"""Bab 6: EM untuk model lain. Campuran Bernoulli multivariat, dan
EM untuk normal bivariat dengan nilai hilang."""
import numpy as np
from scipy.optimize import minimize

from bab01_data import BENIH, data_mini
from bab06_model import B, bern_e, bern_m

np.set_printoptions(precision=4, suppress=True)

# (1) Campuran Bernoulli: 6 responden x 3 butir ya/tidak
pi, th = np.array([0.5, 0.5]), np.array([[0.75] * 3, [0.25] * 3])
print("(1) Campuran Bernoulli, awal theta1 = 3/4, theta2 = 1/4:")
print("     t   log-kemungkinan   r_i1")
f3 = lambda v: " ".join("%.3f" % a for a in v)
for t in range(200):
    R, ll = bern_e(B, pi, th)
    if t in (0, 1, 2, 5, 10, 50):
        print("    %2d   %.6f        %s" % (t, ll, f3(R[:, 0])))
    if t == 0:
        th1 = bern_m(B, R)[1]
    pi, th = bern_m(B, R)
print("    iterasi 1: theta1 =", f3(th1[0]))
print("               theta2 =", f3(th1[1]))
print("    t = 200:   theta1 =", f3(th[0]), " pi =", f3(pi))
print("               theta2 =", f3(th[1]))
print("    log-kemungkinan t = 200: %.6f" % bern_e(B, pi, th)[1])


def nll(v):
    p = 1 / (1 + np.exp(-v[0]))
    t_ = 1 / (1 + np.exp(-v[1:].reshape(2, 3)))
    return -bern_e(B, np.array([p, 1 - p]), np.clip(t_, 1e-12, 1 - 1e-12))[1]


v0 = np.r_[0.0, np.log(3) * np.ones(3), -np.log(3) * np.ones(3)]
opt = minimize(nll, v0, method="BFGS")
print("    BFGS dari awal yang sama: l = %.4f" % -opt.fun)

# (2) Normal bivariat, x2 sampel 4 hilang
X, g = data_mini()
x = X.copy()
x[3, 1] = np.nan
lengkap = ~np.isnan(x[:, 1])
mu = x[lengkap].mean(0)
S = np.cov(x[lengkap].T, bias=True)
print("(2) x2 sampel 4 hilang; awal = kasus lengkap:")
print("    mu =", mu, " Sigma =", S.round(4).tolist())
for t in range(2):
    b1 = S[0, 1] / S[0, 0]
    xh = mu[1] + b1 * (x[3, 0] - mu[0])          # E[x2 | x1]
    v = S[1, 1] - S[0, 1] ** 2 / S[0, 0]         # Var[x2 | x1]
    Z = x.copy()
    Z[3, 1] = xh
    mu = Z.mean(0)
    D = Z - mu
    S = D.T @ D / 6
    S[1, 1] += v / 6
    print("    iterasi %d: isian = %.4f, var bersyarat = %.4f" % (t + 1, xh, v))
    print("      mu =", mu, " Sigma =", S.round(4).tolist())

# (3) Simulasi: 40% x2 hilang secara acak, 2000 ulangan, m = 50
rng = np.random.default_rng(BENIH)
S_b = np.array([[6.0, 3.0], [3.0, 6.0]])
L = np.linalg.cholesky(S_b)
hasil = {"kasus lengkap": [], "isi rata-rata": [], "EM": []}
for _ in range(2000):
    Y = 4 + rng.standard_normal((50, 2)) @ L.T
    # MAR: x2 lebih sering hilang bila x1 besar
    h = rng.random(50) < np.where(Y[:, 0] > 4, 0.7, 0.1)
    o = ~h
    cc = np.cov(Y[o].T, bias=True)
    hasil["kasus lengkap"].append((Y[o, 1].mean(), cc[1, 1]))
    Z = Y.copy()
    Z[h, 1] = Y[o, 1].mean()
    hasil["isi rata-rata"].append((Z[:, 1].mean(),
                                   np.cov(Z.T, bias=True)[1, 1]))
    m_ = Y[o].mean(0)
    m_[0] = Y[:, 0].mean()
    s_ = cc.copy()
    s_[0, 0] = Y[:, 0].var()
    for _ in range(100):
        b1 = s_[0, 1] / s_[0, 0]
        Z = Y.copy()
        Z[h, 1] = m_[1] + b1 * (Y[h, 0] - m_[0])
        v = s_[1, 1] - s_[0, 1] ** 2 / s_[0, 0]
        m_ = Z.mean(0)
        D = Z - m_
        s_ = D.T @ D / 50
        s_[1, 1] += v * h.sum() / 50
    hasil["EM"].append((m_[1], s_[1, 1]))
print("(3) mu2 = 4, Var(x2) = 6; m = 50; x2 hilang dengan peluang")
print("    0.7 bila x1 > 4 dan 0.1 bila tidak; 2000 ulangan:")
print("    %-14s %-14s %s" % ("cara", "rata-rata mu2", "rata-rata Var(x2)"))
for k, v in hasil.items():
    v = np.array(v)
    print("    %-14s %-14.3f %.3f" % (k, v[:, 0].mean(), v[:, 1].mean()))
print("    MLE tanpa nilai hilang: harapan Var = 49/50 * 6 = %.3f"
      % (49 / 50 * 6))
