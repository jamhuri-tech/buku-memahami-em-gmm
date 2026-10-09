"""Bab 5: ketaksamaan Jensen, ELBO, dekomposisi log p = F + KL, dan
kenaikan monoton EM, diperiksa pada data mini."""
import numpy as np

from bab01_data import (awal_mini, data_mini, langkah_m, log_gabung,
                        tanggung_jawab)

np.set_printoptions(precision=4, suppress=True)
X, g = data_mini()
pi0, mu0, S0 = awal_mini()


def F(q, pi, mu, S):
    """ELBO: sum_i sum_k q_ik [log pi_k + log N_ik - log q_ik]."""
    A = log_gabung(X, pi, mu, S)
    qlogq = np.where(q > 0, q * np.log(np.where(q > 0, q, 1)), 0.0)
    return float((q * A).sum() - qlogq.sum())


def Q(q, pi, mu, S):
    return float((q * log_gabung(X, pi, mu, S)).sum())


def H(q):
    return float(-np.where(q > 0, q * np.log(np.where(q > 0, q, 1)), 0).sum())


# (1) Satu sampel: batas Jensen untuk dua pilihan q
A1 = log_gabung(X[:1], pi0, mu0, S0)[0]
lp1 = np.log(np.exp(A1).sum())
r1 = np.exp(A1 - lp1)
print("(1) sampel 1, log p(x_1) = %.4f" % lp1)
for nama, q in [("q = (1/2, 1/2)", np.array([0.5, 0.5])), ("q = r_1", r1)]:
    f = float(q @ A1 - q @ np.log(q))
    print("    %-15s F_1 = %.4f  KL = %.4f" % (nama, f, abs(lp1 - f)))

# (2) Rantai ketaksamaan pada setiap iterasi
print("(2)  t   l(t)       F(r(t), t+1)   l(t+1)    KL")
pi, mu, S = pi0, mu0, S0
for t in range(6):
    R, ll = tanggung_jawab(X, pi, mu, S)
    assert np.isclose(F(R, pi, mu, S), ll)       # langkah E: KL = 0
    pi_b, mu_b, S_b = langkah_m(X, R)
    f_baru = F(R, pi_b, mu_b, S_b)
    ll_baru = tanggung_jawab(X, pi_b, mu_b, S_b)[1]
    print("    %d  %9.4f   %9.4f     %9.4f   %.4f"
          % (t, ll, f_baru, ll_baru, abs(ll_baru - f_baru)))
    if t == 0:
        Q0, Q1, H0 = Q(R, pi, mu, S), Q(R, pi_b, mu_b, S_b), H(R)
    pi, mu, S = pi_b, mu_b, S_b
print("    t = 0: H(r) = %.4f" % H0)
print("           Q(lama) = %.4f  Q(baru) = %.4f" % (Q0, Q1))

# (3) EM satu parameter: pi saja, komponen ditahan pada nilai awal
rho = np.exp(log_gabung(X, np.array([.5, .5]), mu0, S0) @ [1, -1])
print("(3) rho_i = N_i1/N_i2 =", " ".join("%.4g" % v for v in rho))


def ll_pi(p):
    return float(np.log(p * rho + 1 - p).sum())


p = 0.75
for t in range(6):
    r = p * rho / (p * rho + 1 - p)
    print("    t = %d  pi = %.6f  l(pi) - l(1/2) = %.6f"
          % (t, p, ll_pi(p) - ll_pi(0.5)))
    p = r.mean()
