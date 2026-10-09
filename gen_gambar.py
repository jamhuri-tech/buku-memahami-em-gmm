# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20261009  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    """Gaya sumbu seri: tanpa bingkai atas dan kanan."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
#  Gambar per bab ditambahkan di bawah ini sebagai fungsi babNN_nama().





# ============================ Bab 1 ==================================

# ============================ Bab 1 ==================================


# ============================ Bab 1 ==================================

def _mini():
    from bab01_data import data_mini
    return data_mini()


def _titik(ax, X, g, label=True, nomor=None):
    for k, (w, mk) in enumerate([(BIRU, "o"), (JINGGA, "s")], start=1):
        s = g == k
        ax.scatter(X[s, 0], X[s, 1], color=w, marker=mk, s=18, zorder=3,
                   label=f"g = {k}" if label else None)
    nomor = range(1, len(X) + 1) if nomor is None else nomor
    for i, (a, b) in zip(nomor, X):
        ax.annotate(str(i), (a, b), xytext=(3, 3), textcoords="offset points",
                    fontsize=6, color=ABU)


def _elips(ax, mu, S, warna, c=(1, 2), **kw):
    from matplotlib.patches import Ellipse
    val, vec = np.linalg.eigh(S)
    sudut = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    for cc in c:
        ax.add_patch(Ellipse(mu, 2 * cc * np.sqrt(val[1]),
                             2 * cc * np.sqrt(val[0]), angle=sudut,
                             fill=False, color=warna, lw=0.7, **kw))


def _sumbu_mini(ax):
    ax.set_xlim(-1.5, 9.5)
    ax.set_ylim(-1.5, 9.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$ (kali per bulan)")
    ax.set_ylabel("$x_2$ (ratus ribu Rp)")
    _rapikan(ax)


def bab01_data():
    X, g = _mini()
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.4))
    for a in ax:
        _titik(a, X, g)
        _sumbu_mini(a)
    mu = X.mean(0)
    S = (X - mu).T @ (X - mu) / len(X)
    _elips(ax[0], mu, S, ABU)
    ax[0].plot(*mu, marker="*", color=MERAH, ms=7, zorder=4)
    ax[0].set_title("satu Gaussian")
    for k, w in [(1, BIRU), (2, JINGGA)]:
        Xk = X[g == k]
        mk = Xk.mean(0)
        _elips(ax[1], mk, (Xk - mk).T @ (Xk - mk) / 3, w)
        ax[1].plot(*mk, marker="*", color=w, ms=7, zorder=4)
    ax[1].set_title("dua komponen")
    ax[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    simpan(fig, "bab01-data")


def bab01_kepadatan():
    from bab01_data import log_normal, em, awal_mini
    X, g = _mini()
    a = np.linspace(-1.5, 9.5, 220)
    G = np.array(np.meshgrid(a, a)).reshape(2, -1).T
    mu = X.mean(0)
    S = (X - mu).T @ (X - mu) / len(X)
    p1 = np.exp(log_normal(G, mu, S))
    pi, m2, S2, _ = em(X, *awal_mini())
    p2 = sum(pi[k] * np.exp(log_normal(G, m2[k], S2[k])) for k in range(2))
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.4))
    for a_, p, j in [(ax[0], p1, "$p(\\mathbf{x})$ satu Gaussian"),
                     (ax[1], p2, "$p(\\mathbf{x})$ campuran $K = 2$")]:
        a_.contourf(a, a, p.reshape(len(a), len(a)), levels=12,
                    cmap="Blues")
        _titik(a_, X, g, label=False)
        _sumbu_mini(a_)
        a_.set_title(j)
    fig.tight_layout()
    simpan(fig, "bab01-kepadatan")


def bab01_alur():
    fig, ax = plt.subplots(figsize=(4.7, 2.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    gaya = dict(ha="center", va="center", fontsize=6.5, color=BIRU)

    def kotak(x, y, t, warna=BIRU_MUDA):
        return ax.text(x, y, t, bbox=dict(boxstyle="round,pad=0.4",
                                          fc=warna, ec=ABU_GARIS, lw=0.6),
                       **gaya)

    k = [kotak(0.6, 3.3, "$\\mathbf{x}_i$"),
         kotak(2.6, 3.3, "$\\log\\mathcal{N}_{ik}$"),
         kotak(5.2, 3.3, "$a_{ik} = \\log\\pi_k + \\log\\mathcal{N}_{ik}$"),
         kotak(8.6, 3.3, "$\\ell_i = \\log\\sum_k e^{a_{ik}}$"),
         kotak(8.6, 1.9, "$r_{ik} = e^{a_{ik} - \\ell_i}$"),
         kotak(4.6, 1.9, "$\\partial\\ell_i/\\partial\\mu_k,\\;"
                         "\\partial\\ell_i/\\partial\\Sigma_k,\\;"
                         "\\partial\\ell_i/\\partial\\beta_k$"),
         kotak(4.6, 0.5, "$\\sum_i \\partial\\ell_i = 0$:  "
                         "$\\pi_k, \\mu_k, \\Sigma_k$ baru", JINGGA_MUDA)]
    fig.canvas.draw()
    sambung = [(0, 1, (1, .5), (0, .5)), (1, 2, (1, .5), (0, .5)),
               (2, 3, (1, .5), (0, .5)), (3, 4, (.5, 0), (.5, 1)),
               (4, 5, (0, .5), (1, .5)), (5, 6, (.5, 0), (.5, 1))]
    for a, b, pa, pb in sambung:
        ax.annotate("", xy=pb, xycoords=k[b], xytext=pa, textcoords=k[a],
                    arrowprops=dict(arrowstyle="->", color=ABU, lw=0.7,
                                    shrinkA=3, shrinkB=3))
    ax.text(0.6, 2.75, "satu sampel", ha="center", fontsize=6, color=ABU)
    ax.text(5.2, 2.6, "langkah E: tanggung jawab", ha="center",
            fontsize=6, color=HIJAU)
    ax.text(8.0, 0.5, "langkah M", ha="center", fontsize=6.5,
            color=JINGGA)
    simpan(fig, "bab01-alur")


# ============================ Bab 2 ==================================

def bab02_transformasi():
    rng = np.random.default_rng(BENIH)
    Z = rng.standard_normal((400, 2))
    S = np.array([[6.0, 3.0], [3.0, 6.0]])
    mu = np.array([4.0, 4.0])
    L = np.linalg.cholesky(S)
    Xs = mu + Z @ L.T
    t = np.linspace(0, 2 * np.pi, 200)
    lingk = np.column_stack([np.cos(t), np.sin(t)])
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.4))
    ax[0].scatter(Z[:, 0], Z[:, 1], s=2, color=ABU_GARIS)
    for c in (1, 2):
        ax[0].plot(c * lingk[:, 0], c * lingk[:, 1], color=BIRU, lw=0.7)
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-4, 4)
    ax[0].set_title("$\\mathbf{z} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$")
    ax[0].set_xlabel("$z_1$")
    ax[0].set_ylabel("$z_2$")
    ax[1].scatter(Xs[:, 0], Xs[:, 1], s=2, color=ABU_GARIS)
    for c in (1, 2):
        e = mu + c * lingk @ L.T
        ax[1].plot(e[:, 0], e[:, 1], color=BIRU, lw=0.7)
    val, vec = np.linalg.eigh(S)
    for j, w in [(1, MERAH), (0, HIJAU)]:
        u = vec[:, j] * np.sqrt(val[j])
        ax[1].annotate("", xy=mu + u, xytext=mu,
                       arrowprops=dict(arrowstyle="->", color=w, lw=1))
        ax[1].text(*(mu + 1.6 * u), f"$\\sqrt{{\\lambda_{2 - j}}}\\,"
                   f"\\mathbf{{q}}_{2 - j}$", color=w, fontsize=6.5,
                   ha="center", va="center")
    ax[1].set_xlim(-6, 14)
    ax[1].set_ylim(-6, 14)
    ax[1].set_title("$\\mathbf{x} = \\boldsymbol{\\mu} + \\mathbf{L}\\mathbf{z}$")
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")
    for a in ax:
        a.set_aspect("equal")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab02-transformasi")


def bab02_mahalanobis():
    X, g = _mini()
    mu = X.mean(0)
    S = (X - mu).T @ (X - mu) / len(X)
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    t = np.linspace(0, 2 * np.pi, 300)
    L = np.linalg.cholesky(S)
    for c, gaya in [(1, ":"), (np.sqrt(2), "-"), (2, ":")]:
        e = mu + c * np.column_stack([np.cos(t), np.sin(t)]) @ L.T
        ax.plot(e[:, 0], e[:, 1], color=BIRU if c == np.sqrt(2) else ABU,
                lw=0.8, ls=gaya)
    val, vec = np.linalg.eigh(S)
    for j in (0, 1):
        u = vec[:, j] * np.sqrt(2 * val[j])
        ax.plot([mu[0] - u[0], mu[0] + u[0]], [mu[1] - u[1], mu[1] + u[1]],
                color=ABU, lw=0.5, ls="--")
    _titik(ax, X, g, label=False)
    ax.plot(*mu, marker="*", color=MERAH, ms=7, zorder=4)
    ax.text(7.6, 5.6, "$d^2 = 2$", color=BIRU, fontsize=6.5)
    _sumbu_mini(ax)
    fig.tight_layout()
    simpan(fig, "bab02-mahalanobis")


def bab02_chi2():
    from scipy import stats
    rng = np.random.default_rng(BENIH)
    S = np.array([[2.0, -1.0], [-1.0, 2.0]])
    L = np.linalg.cholesky(S)
    Z = rng.standard_normal((20000, 2)) @ L.T
    d2 = np.einsum("ij,jk,ik->i", Z, np.linalg.inv(S), Z)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.0))
    ax[0].hist(d2, bins=np.linspace(0, 12, 49), density=True,
               color=BIRU_MUDA, edgecolor=ABU_GARIS, lw=0.3)
    u = np.linspace(0, 12, 200)
    ax[0].plot(u, stats.chi2.pdf(u, 2), color=BIRU, lw=0.9,
               label="$\\chi^2_2$: $\\frac{1}{2} e^{-u/2}$")
    ax[0].axvline(stats.chi2.ppf(0.95, 2), color=MERAH, lw=0.7, ls="--")
    ax[0].set_xlabel("$d^2$")
    ax[0].legend(fontsize=6)
    mm = np.array([4, 6, 10, 20, 50, 100, 200])
    cak = []
    for k in mm:
        hit = 0
        for _ in range(2000):
            A = rng.standard_normal((k, 2)) @ L.T
            mA = A.mean(0)
            SA = (A - mA).T @ (A - mA) / k
            xb = rng.standard_normal(2) @ L.T
            hit += (xb - mA) @ np.linalg.solve(SA, xb - mA) <= 5.9915
        cak.append(hit / 2000)
    ax[1].plot(mm, cak, marker="o", ms=2.5, color=JINGGA, lw=0.8)
    ax[1].axhline(0.95, color=MERAH, lw=0.7, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xticks(mm)
    ax[1].set_xticklabels([str(v) for v in mm])
    kunci_label(ax[1], "x")
    ax[1].set_xlabel("$m$ (ukuran sampel taksiran)")
    ax[1].set_ylabel("cakupan elips 95%")
    for a in ax:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab02-chi2")


# ============================ Bab 3 ==================================

def bab03_generatif():
    from bab01_data import akhir_mini, bangkitkan
    rng = np.random.default_rng(BENIH)
    X, z = bangkitkan(500, *akhir_mini(), rng)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.4))
    ax[0].scatter(X[:, 0], X[:, 1], s=2, color=ABU)
    ax[0].set_title("yang teramati: $\\mathbf{x}_i$")
    for k, w in [(0, BIRU), (1, JINGGA)]:
        ax[1].scatter(X[z == k, 0], X[z == k, 1], s=2, color=w,
                      label=f"$z_i = {k + 1}$")
    ax[1].set_title("dengan variabel laten $z_i$")
    ax[1].legend(fontsize=6, loc="upper left", markerscale=3)
    for a in ax:
        a.set_xlim(-2.5, 10.5)
        a.set_ylim(-2.5, 10.5)
        a.set_aspect("equal")
        a.set_xlabel("$x_1$")
        a.set_ylabel("$x_2$")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab03-generatif")


def bab03_permukaan():
    from bab01_data import data_mini, tanggung_jawab
    X, _ = data_mini()
    S = np.array([[2.0, -1.0], [-1.0, 2.0]])
    a = np.linspace(-1, 9, 161)
    Z = np.empty((len(a), len(a)))
    for i, u in enumerate(a):
        for j, v in enumerate(a):
            mu = np.array([[u, u], [v, v]])
            Z[j, i] = tanggung_jawab(X, np.array([.5, .5]), mu,
                                     np.array([S, S]))[1]
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    cs = ax.contourf(a, a, np.maximum(Z, -60), levels=20, cmap="Blues")
    ax.contour(a, a, Z, levels=[-30, -27, -25], colors=ABU, linewidths=0.4)
    ax.plot([2, 6], [6, 2], "*", color=MERAH, ms=7)
    ax.plot([-1, 9], [-1, 9], ls=":", color=ABU, lw=0.6)
    ax.set_xlabel("$\\mu_1 = (a, a)$:  $a$")
    ax.set_ylabel("$\\mu_2 = (b, b)$:  $b$")
    ax.set_aspect("equal")
    _rapikan(ax)
    fig.colorbar(cs, ax=ax, shrink=0.8).ax.tick_params(labelsize=6)
    fig.tight_layout()
    simpan(fig, "bab03-permukaan")


def bab03_singular():
    from bab01_data import data_mini, tanggung_jawab
    X, _ = data_mini()
    mu_s = X.mean(0)
    S_s = (X - mu_s).T @ (X - mu_s) / 6
    eps = np.logspace(-8, 1, 120)
    ll = [tanggung_jawab(X, np.array([.5, .5]), np.array([X[0], mu_s]),
                         np.array([e * np.eye(2), S_s]))[1] for e in eps]
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    ax.plot(eps, ll, color=BIRU, lw=0.9,
            label="$\\mu_1 = \\mathbf{x}_1$, $\\Sigma_1 = \\varepsilon\\mathbf{I}$")
    ax.axhline(-24.482, color=MERAH, lw=0.7, ls="--",
               label="hasil EM ($-24{,}48$)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("$\\varepsilon$")
    ax.set_ylabel("log-kemungkinan")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-singular")


# ============================ Bab 4 ==================================

def bab04_iterasi():
    from bab01_data import awal_mini, tanggung_jawab, langkah_m
    X, g = _mini()
    pi, mu, S = awal_mini()
    keadaan = []
    for t in range(6):
        R, ll = tanggung_jawab(X, pi, mu, S)
        keadaan.append((t, pi, mu, S, R, ll))
        pi, mu, S = langkah_m(X, R)
    pilih = [0, 1, 3, 5]
    fig, ax = plt.subplots(1, 4, figsize=(4.7, 1.55))
    for a, t in zip(ax, pilih):
        _, p_, m_, S_, R, ll = keadaan[t]
        warna = [tuple(np.array(matplotlib.colors.to_rgb(BIRU)) * r +
                       np.array(matplotlib.colors.to_rgb(JINGGA)) * (1 - r))
                 for r in R[:, 0]]
        a.scatter(X[:, 0], X[:, 1], c=warna, s=10, zorder=3)
        for k, w in [(0, BIRU), (1, JINGGA)]:
            _elips(a, m_[k], S_[k], w, c=(1,))
            a.plot(*m_[k], marker="+", color=w, ms=5)
        a.set_xlim(-2, 10)
        a.set_ylim(-2, 10)
        a.set_aspect("equal")
        a.set_xticks([0, 4, 8])
        a.set_yticks([0, 4, 8])
        a.tick_params(labelsize=5.5)
        a.set_title(f"$t = {t}$, $\\ell = {angka_mat(ll, 2)}$", fontsize=6.5)
        _rapikan(a)
    fig.tight_layout(w_pad=0.3)
    simpan(fig, "bab04-iterasi")


def bab04_riwayat():
    from bab01_data import awal_mini, em
    X, _ = _mini()
    _, _, _, riw = em(X, *awal_mini(), iterasi=50, tol=1e-12)
    riw = np.array(riw)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.9))
    ax[0].plot(range(len(riw)), riw, marker="o", ms=2.5, color=BIRU, lw=0.8)
    ax[0].set_xlabel("iterasi $t$")
    ax[0].set_ylabel("log-kemungkinan")
    sel = riw[-1] - riw[:-1]
    ax[1].semilogy(range(len(sel)), np.maximum(sel, 1e-16), marker="o",
                   ms=2.5, color=JINGGA, lw=0.8)
    ax[1].set_xlabel("iterasi $t$")
    ax[1].set_ylabel("$\\ell^\\ast - \\ell^{(t)}$")
    for a in ax:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab04-riwayat")


def bab04_besar():
    from bab01_data import bangkitkan, em
    rng = np.random.default_rng(BENIH)
    pi_b = np.array([0.5, 0.3, 0.2])
    mu_b = np.array([[0.0, 0.0], [5.0, 1.0], [2.0, 5.0]])
    S_b = np.array([[[2.0, 0.8], [0.8, 1.0]], [[1.0, -0.6], [-0.6, 1.5]],
                    [[0.5, 0.0], [0.0, 2.0]]])
    Xb, zb = bangkitkan(900, pi_b, mu_b, S_b, rng)
    awal = Xb[rng.choice(900, 3, replace=False)]
    c = np.cov(Xb.T, bias=True)
    pi_h, mu_h, S_h, _ = em(Xb, np.full(3, 1 / 3), awal, np.array([c] * 3),
                            iterasi=1000, tol=1e-10)
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.scatter(Xb[:, 0], Xb[:, 1], s=1.5, color=ABU_GARIS)
    for k in range(3):
        _elips(ax, mu_b[k], S_b[k], ABU, c=(2,), ls="--")
        _elips(ax, mu_h[k], S_h[k], [BIRU, JINGGA, HIJAU][k], c=(2,))
    ax.plot(awal[:, 0], awal[:, 1], "x", color=MERAH, ms=5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-besar")


# ============================ Bab 5 ==================================

def bab05_jensen():
    a = np.array([1.0, 7.0])
    q = np.array([0.75, 0.25])
    u = np.linspace(0.6, 8, 200)
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    ax.plot(u, np.log(u), color=BIRU, lw=0.9, label="$\\log u$")
    ax.plot(a, np.log(a), color=ABU, lw=0.7, ls="--", label="tali busur")
    ea = q @ a
    ax.plot([ea], [np.log(ea)], "o", color=BIRU, ms=4)
    ax.plot([ea], [q @ np.log(a)], "o", color=JINGGA, ms=4)
    ax.plot(a, np.log(a), "o", color=ABU, ms=3)
    ax.annotate("$\\log \\sum_k q_k a_k$", (ea, np.log(ea)), xytext=(-62, 8),
                textcoords="offset points", fontsize=6.5, color=BIRU)
    ax.annotate("$\\sum_k q_k \\log a_k$", (ea, q @ np.log(a)),
                xytext=(6, -12), textcoords="offset points", fontsize=6.5,
                color=JINGGA)
    ax.vlines(ea, q @ np.log(a), np.log(ea), color=MERAH, lw=0.8)
    ax.text(ea + 0.15, (q @ np.log(a) + np.log(ea)) / 2, "celah", fontsize=6,
            color=MERAH)
    ax.set_xlabel("$u$")
    ax.legend(fontsize=6, loc="lower right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-jensen")


def bab05_batas():
    from bab01_data import awal_mini, log_gabung
    X, _ = _mini()
    _, mu0, S0 = awal_mini()
    p = np.linspace(0.3, 0.95, 300)

    def A(pp):
        return log_gabung(X, np.array([pp, 1 - pp]), mu0, S0)

    ll = np.array([np.log(np.exp(A(v)).sum(1)).sum() for v in p])
    fig, ax = plt.subplots(figsize=(4.7, 2.2))
    ax.plot(p, ll, color=BIRU, lw=1.0, label="$\\ell(\\pi_1)$")
    pt = 0.75
    for t, w in [(0, JINGGA), (1, HIJAU)]:
        At = A(pt)
        R = np.exp(At - np.log(np.exp(At).sum(1, keepdims=True)))
        H = -(R * np.log(R)).sum()
        Fp = np.array([(R * A(v)).sum() + H for v in p])
        ax.plot(p, Fp, color=w, lw=0.8, ls="--",
                label=f"$F(r^{{({t})}}, \\pi_1)$")
        ax.plot([pt], [np.log(np.exp(At).sum(1)).sum()], "o", color=w, ms=3.5)
        baru = R[:, 0].mean()
        ax.plot([baru], [Fp[np.argmin(abs(p - baru))]], "^", color=w, ms=4)
        ax.annotate("", xy=(baru, ll[np.argmin(abs(p - baru))]),
                    xytext=(baru, Fp[np.argmin(abs(p - baru))]),
                    arrowprops=dict(arrowstyle="->", color=w, lw=0.6))
        pt = baru
    ax.set_ylim(ll.min() - 0.6, ll.max() + 0.3)
    ax.set_xlabel("$\\pi_1$")
    ax.set_ylabel("log-kemungkinan")
    ax.legend(fontsize=6, loc="lower left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-batas")


def bab05_rantai():
    from bab01_data import awal_mini, tanggung_jawab, langkah_m, log_gabung
    X, _ = _mini()
    pi, mu, S = awal_mini()
    ll, Fb = [], []
    for t in range(6):
        R, l = tanggung_jawab(X, pi, mu, S)
        ll.append(l)
        pi, mu, S = langkah_m(X, R)
        A = log_gabung(X, pi, mu, S)
        Fb.append((R * A).sum() - (R * np.log(R)).sum())
    ll.append(tanggung_jawab(X, pi, mu, S)[1])
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    t = np.arange(len(ll))
    ax.plot(t, ll, marker="o", ms=3, color=BIRU, lw=0.8,
            label="$\\ell(\\theta^{(t)})$")
    ax.plot(t[1:], Fb, marker="^", ms=3, color=JINGGA, lw=0, 
            label="$F(r^{(t-1)}, \\theta^{(t)})$")
    for k in range(len(Fb)):
        ax.vlines(k + 1, Fb[k], ll[k + 1], color=MERAH, lw=0.8)
        ax.hlines(ll[k], k, k + 1, color=ABU_GARIS, lw=0.6, ls=":")
    ax.set_xlabel("iterasi $t$")
    ax.set_ylabel("log-kemungkinan")
    ax.legend(fontsize=6, loc="lower right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-rantai")


# ============================ Bab 6 ==================================

def bab06_bernoulli():
    import bab06_model as b6
    B = b6.B
    pi, th = np.array([0.5, 0.5]), np.array([[0.75] * 3, [0.25] * 3])
    jejak, lls = [], []
    for t in range(40):
        R, ll = b6.bern_e(B, pi, th)
        jejak.append(th.copy())
        lls.append(ll)
        pi, th = b6.bern_m(B, R)
    jejak = np.array(jejak)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.9))
    nama = ["daring", "promo", "poin"]
    for j, w in enumerate([BIRU, JINGGA, HIJAU]):
        ax[0].plot(jejak[:, 0, j], color=w, lw=0.9, label=nama[j])
        ax[0].plot(jejak[:, 1, j], color=w, lw=0.9, ls="--")
    ax[0].set_xlabel("iterasi $t$")
    ax[0].set_ylabel("$\\theta_{kj}$")
    ax[0].legend(fontsize=5.5, ncol=3, loc="center right")
    ax[1].plot(lls, color=BIRU, lw=0.9)
    ax[1].axhline(-11.797053, color=MERAH, lw=0.6, ls=":")
    ax[1].set_xlabel("iterasi $t$")
    ax[1].set_ylabel("log-kemungkinan")
    for a in ax:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab06-bernoulli")


def bab06_isian():
    X, g = _mini()
    u = np.linspace(-3, 11, 400)
    f1 = np.exp(-(u - 3.4) ** 2 / (2 * 3.24)) / np.sqrt(2 * np.pi * 3.24)
    f2 = 0.5 * (np.exp(-(u - 1) ** 2 / 3) + np.exp(-(u - 7) ** 2 / 3)) \
        / np.sqrt(2 * np.pi * 1.5)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.3),
                           gridspec_kw=dict(width_ratios=[1, 1.3]))
    lain = np.arange(6) != 3
    _titik(ax[0], X[lain], g[lain], label=False, nomor=[1, 2, 3, 5, 6])
    ax[0].axvline(4, color=ABU, lw=0.6, ls=":")
    ax[0].plot(4, 7, marker="s", mfc="none", mec=JINGGA, ms=5)
    ax[0].text(4.3, 5.9, "4: $x_2$ hilang", fontsize=6, color=JINGGA)
    _sumbu_mini(ax[0])
    ax[1].plot(u, f1, color=ABU, lw=0.9, label="satu Gaussian")
    ax[1].plot(u, f2, color=BIRU, lw=0.9, label="GMM $K = 2$")
    ax[1].axvline(7, color=JINGGA, lw=0.7, ls="--", label="nilai sebenarnya")
    ax[1].set_xlabel("$x_2 \\mid x_1 = 4$")
    ax[1].set_ylabel("kepadatan")
    ax[1].legend(fontsize=5.5, loc="upper left")
    _rapikan(ax[1])
    fig.tight_layout()
    simpan(fig, "bab06-isian")


# ============================ Bab 7 ==================================

def bab07_laju():
    from bab01_data import awal_mini, log_gabung, em
    X, _ = _mini()
    _, mu0, S0 = awal_mini()
    rho = np.exp(log_gabung(X, np.array([.5, .5]), mu0, S0) @ [1, -1])
    p, jar = 0.75, []
    for _ in range(25):
        jar.append(abs(p - 0.5))
        p = np.mean(p * rho / (p * rho + 1 - p))
    _, _, _, riw = em(X, *awal_mini(), iterasi=50, tol=1e-14)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.9))
    t = np.arange(len(jar))
    ax[0].semilogy(t, jar, "o", ms=2.5, color=BIRU, label="EM")
    ax[0].semilogy(t, 0.25 * 0.62 ** t, color=MERAH, lw=0.7, ls="--",
                   label="$0{,}25 \\times 0{,}62^t$")
    ax[0].set_xlabel("iterasi $t$")
    ax[0].set_ylabel("$|\\pi^{(t)} - \\frac{1}{2}|$")
    ax[0].set_title("$\\pi$ saja: laju linear 0,62", fontsize=7)
    ax[0].legend(fontsize=6)
    sel = np.array(riw[-1]) - np.array(riw[:-1])
    ax[1].semilogy(np.arange(len(sel)), np.maximum(sel, 1e-16), "o-", ms=2.5,
                   color=JINGGA, lw=0.7)
    ax[1].set_xlabel("iterasi $t$")
    ax[1].set_ylabel("$\\ell^\\ast - \\ell^{(t)}$")
    ax[1].set_title("GMM data mini", fontsize=7)
    for a in ax:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab07-laju")


def bab07_pelana():
    from bab01_data import SIGMA0_2, tanggung_jawab, langkah_m
    X, g = _mini()
    S0 = np.array([SIGMA0_2 * np.eye(2)] * 2)
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.75))
    for a, e, judul in zip(ax, (0.0, 0.01, -0.01),
                           ("$\\mu_1 = (3;\\,4)$", "$\\mu_1 = (3;\\,4{,}01)$",
                            "$\\mu_1 = (3;\\,3{,}99)$")):
        pi, mu, S = np.array([.5, .5]), np.array([[3., 4 + e], [5., 4.]]), S0
        jalur = [mu.copy()]
        for _ in range(25 if e == 0 else 60):
            R, _ = tanggung_jawab(X, pi, mu, S)
            pi, mu, S = langkah_m(X, R)
            jalur.append(mu.copy())
        jalur = np.array(jalur)
        warna = [tuple(np.array(matplotlib.colors.to_rgb(BIRU)) * r +
                       np.array(matplotlib.colors.to_rgb(JINGGA)) * (1 - r))
                 for r in R[:, 0]]
        a.scatter(X[:, 0], X[:, 1], c=warna, s=10, zorder=3)
        for k, w in [(0, BIRU), (1, JINGGA)]:
            a.plot(jalur[:, k, 0], jalur[:, k, 1], color=w, lw=0.7)
            _elips(a, mu[k], S[k], w, c=(1,))
        a.set_xlim(-1, 9)
        a.set_ylim(-1, 9)
        a.set_aspect("equal")
        a.set_xticks([0, 4, 8])
        a.set_yticks([0, 4, 8])
        a.tick_params(labelsize=5.5)
        a.set_title(judul, fontsize=6.5)
        _rapikan(a)
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab07-pelana")


def bab07_pemulaan():
    import warnings
    from sklearn.mixture import GaussianMixture
    from sklearn.exceptions import ConvergenceWarning
    from bab01_data import bangkitkan
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(BENIH)
    mu_b = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], float)
    Xb, _ = bangkitkan(400, np.array([.25] * 4), mu_b,
                       np.array([np.eye(2) * 1.2] * 4), rng)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.0),
                           gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax[0].scatter(Xb[:, 0], Xb[:, 1], s=1.5, color=ABU)
    ax[0].set_aspect("equal")
    ax[0].set_title("data 400 sampel", fontsize=7)
    cara = ["random", "random_from_data", "k-means++", "kmeans"]
    hasil = []
    for c in cara:
        hasil.append([GaussianMixture(4, init_params=c, n_init=1,
                                      max_iter=500, random_state=s)
                      .fit(Xb).score(Xb) * 400 for s in range(200)])
    bp = ax[1].boxplot(hasil, vert=False, widths=0.5, patch_artist=True,
                       flierprops=dict(ms=2, mec=ABU))
    for b in bp["boxes"]:
        b.set(facecolor=BIRU_MUDA, edgecolor=BIRU, lw=0.6)
    ax[1].set_yticks(range(1, 5))
    ax[1].set_yticklabels(["random", "random\\_from\\_data".replace("\\", ""),
                           "k-means++", "kmeans"], fontsize=6)
    kunci_label(ax[1], "y")
    ax[1].set_xlabel("log-kemungkinan akhir (200 benih)")
    for a in ax:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab07-pemulaan")


# ============================ Bab 8 ==================================

def _kov_penuh(gm, k):
    c = gm.covariances_
    j = gm.covariance_type
    if j == "full":
        return c[k]
    if j == "tied":
        return c
    if j == "diag":
        return np.diag(c[k])
    return c[k] * np.eye(gm.means_.shape[1])


def bab08_struktur():
    import warnings
    from sklearn.mixture import GaussianMixture
    from sklearn.exceptions import ConvergenceWarning
    from bab01_data import bangkitkan
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(BENIH)
    S_a = np.array([[4.0, 3.6], [3.6, 4.0]])
    S_b = np.array([[4.0, -3.6], [-3.6, 4.0]])
    Xs, zs = bangkitkan(600, np.array([0.5, 0.5]),
                        np.array([[0., 0.], [3., 0.]]), np.array([S_a, S_b]),
                        rng)
    fig, ax = plt.subplots(1, 4, figsize=(4.7, 1.6))
    for a, j in zip(ax, ("full", "tied", "diag", "spherical")):
        gm = GaussianMixture(2, covariance_type=j, n_init=10,
                             random_state=BENIH).fit(Xs)
        lab = gm.predict(Xs)
        a.scatter(Xs[:, 0], Xs[:, 1], s=0.8,
                  c=np.where(lab == 0, BIRU, JINGGA))
        for k in range(2):
            _elips(a, gm.means_[k], _kov_penuh(gm, k), MERAH, c=(2,))
        a.set_title(j, fontsize=6.5)
        a.set_xlim(-7, 10)
        a.set_ylim(-8, 8)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        _rapikan(a)
    fig.tight_layout(w_pad=0.3)
    simpan(fig, "bab08-struktur")


def bab08_kmeans():
    s2 = np.logspace(-1.5, 1, 200)
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    for d, w, t in [(12, BIRU, "sampel 3: $\\Delta = 12$"),
                    (24, JINGGA, "sampel 1: $\\Delta = 24$"),
                    (4, HIJAU, "$\\Delta = 4$")]:
        ax.semilogx(s2, 1 / (1 + np.exp(-d / (2 * s2))), color=w, lw=0.9,
                    label=t)
    ax.axvline(6 / np.log(3), color=ABU, lw=0.6, ls=":")
    ax.text(6 / np.log(3) * 1.05, 0.55, "$\\sigma_0^2$", fontsize=6.5,
            color=ABU)
    ax.set_xlabel("$\\sigma^2$")
    ax.set_ylabel("$r_{i1}$")
    ax.legend(fontsize=6, loc="lower left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-kmeans")


def bab08_mini():
    import warnings
    from sklearn.mixture import GaussianMixture
    warnings.filterwarnings("ignore")
    X, g = _mini()
    mu0 = np.array([X[g == k].mean(0) for k in (1, 2)])
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.3))
    for a, j in zip(ax, ("full", "diag")):
        gm = GaussianMixture(2, covariance_type=j, means_init=mu0, tol=1e-12,
                             max_iter=2000, reg_covar=1e-9).fit(X)
        _titik(a, X, g, label=False)
        for k, w in [(0, BIRU), (1, JINGGA)]:
            _elips(a, gm.means_[k], _kov_penuh(gm, k), w)
        a.set_title(f"{j}: $\\ell = {angka_mat(gm.score(X) * 6, 2)}$",
                    fontsize=7)
        _sumbu_mini(a)
    fig.tight_layout()
    simpan(fig, "bab08-mini")


# ============================ Bab 9 ==================================

def bab09_kriteria():
    import warnings
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import KFold
    from bab01_data import bangkitkan
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH + 1)
    mu_b = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], float)
    Xb, _ = bangkitkan(400, np.full(4, 0.25), mu_b,
                       np.array([np.eye(2) * 1.2] * 4), rng)
    Ks = np.arange(1, 9)
    aic, bic, icl, cv, ll = [], [], [], [], []
    for K in Ks:
        gm = GaussianMixture(K, n_init=3, random_state=0).fit(Xb)
        R = gm.predict_proba(Xb)
        ll.append(-2 * gm.score(Xb) * 400)
        aic.append(gm.aic(Xb))
        bic.append(gm.bic(Xb))
        icl.append(gm.bic(Xb) - 2 * np.sum(R * np.log(np.clip(R, 1e-300, 1))))
        c = 0.0
        for a, b in KFold(5, shuffle=True, random_state=0).split(Xb):
            c -= 2 * GaussianMixture(K, random_state=0).fit(Xb[a]).score(
                Xb[b]) * len(b)
        cv.append(c)
    fig, ax = plt.subplots(figsize=(4.7, 2.2))
    for y, w, t, mk in [(ll, ABU, "$-2\\ell$", "."), (aic, BIRU, "AIC", "o"),
                        (bic, JINGGA, "BIC", "s"), (icl, HIJAU, "ICL", "^"),
                        (cv, MERAH, "$-2\\ell$ validasi silang", "d")]:
        ax.plot(Ks, y, marker=mk, ms=3, color=w, lw=0.8, label=t)
        if t != "$-2\\ell$":
            k = int(np.argmin(y))
            ax.plot(Ks[k], y[k], marker=mk, ms=6, mfc="none", mec=w)
    ax.set_xlabel("banyaknya komponen $K$")
    ax.set_ylabel("nilai kriteria")
    ax.legend(fontsize=5.5, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-kriteria")


def bab09_lrt():
    import warnings
    from scipy import stats
    from sklearn.mixture import GaussianMixture
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH)
    lrt = []
    for _ in range(300):
        Y = rng.standard_normal((100, 2))
        l1 = GaussianMixture(1).fit(Y).score(Y) * 100
        l2 = GaussianMixture(2, n_init=5, random_state=0).fit(Y).score(Y) * 100
        lrt.append(2 * (l2 - l1))
    u = np.linspace(0, 35, 300)
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    ax.hist(lrt, bins=np.arange(0, 36, 1.5), density=True, color=BIRU_MUDA,
            edgecolor=ABU_GARIS, lw=0.3, label="simulasi $H_0$")
    ax.plot(u, stats.chi2.pdf(u, 6), color=BIRU, lw=0.9, label="$\\chi^2_6$")
    ax.axvline(stats.chi2.ppf(0.95, 6), color=BIRU, lw=0.6, ls="--")
    ax.axvline(np.quantile(lrt, 0.95), color=MERAH, lw=0.6, ls="--")
    ax.set_xlabel("$2(\\ell_2 - \\ell_1)$")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-lrt")


def bab09_laplace():
    th = np.linspace(-1.5, 2.5, 400)
    fig, ax = plt.subplots(figsize=(4.7, 1.8))
    for m, w in [(10, BIRU), (100, JINGGA)]:
        f = np.exp(-m * (th - 0.5) ** 2 / 2)
        ax.plot(th, f, color=w, lw=0.9,
                label=f"$m = {m}$: lebar $\\propto 1/\\sqrt{{m}}$")
        ax.fill_between(th, f, color=w, alpha=0.12, lw=0)
    ax.plot(th, np.full_like(th, 0.25), color=ABU, lw=0.7, ls=":",
            label="prior $p(\\theta)$ (skala)")
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$e^{\\ell(\\theta) - \\hat\\ell}$")
    ax.legend(fontsize=6, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-laplace")


# ============================ Bab 10 =================================

def bab10_runtuh():
    from bab10_singular import X7, em_map, pi0, mu0, S0
    import io
    import contextlib
    X, g = _mini()
    fig, ax = plt.subplots(1, 4, figsize=(4.7, 1.55))
    for a, t in zip(ax, (0, 1, 2, 3)):
        with contextlib.redirect_stdout(io.StringIO()):
            pi, mu, S, lp = em_map(X7, pi0, mu0, S0, iterasi=t) if t else \
                (pi0, mu0, S0, None)
        a.scatter(X7[:, 0], X7[:, 1], s=8, color=ABU, zorder=3)
        a.scatter([1], [1], s=26, facecolors="none", edgecolors=MERAH, lw=0.7)
        for k, w in [(0, MERAH), (1, BIRU), (2, JINGGA)]:
            _elips(a, mu[k], S[k], w, c=(1,))
        a.set_xlim(-2, 10)
        a.set_ylim(-2, 10)
        a.set_aspect("equal")
        a.set_xticks([0, 4, 8])
        a.set_yticks([0, 4, 8])
        a.tick_params(labelsize=5.5)
        a.set_title(f"$t = {t}$", fontsize=6.5)
        _rapikan(a)
    fig.tight_layout(w_pad=0.3)
    simpan(fig, "bab10-runtuh")


def bab10_map():
    from bab10_singular import X7, em_map, pi0, mu0, S0
    pi, mu, S, _ = em_map(X7, pi0, mu0, S0, alpha=2.0,
                          Psi=0.5 * np.eye(2), nu=1.0, iterasi=500)
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.scatter(X7[:, 0], X7[:, 1], s=10, color=ABU, zorder=3)
    ax.scatter([1], [1], s=34, facecolors="none", edgecolors=MERAH, lw=0.7)
    for k, w in [(0, MERAH), (1, BIRU), (2, JINGGA)]:
        _elips(ax, mu[k], S[k], w)
        ax.plot(*mu[k], "+", color=w, ms=5)
    _sumbu_mini(ax)
    fig.tight_layout()
    simpan(fig, "bab10-map")


def bab10_reg():
    import warnings
    from sklearn.mixture import GaussianMixture
    from bab01_data import bangkitkan
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH)
    n = 6
    mu_b = np.array([np.zeros(n), np.r_[3, np.zeros(n - 1)],
                     np.r_[0, 3, np.zeros(n - 2)]])
    A = rng.standard_normal((n, n)) * 0.3
    S_b = np.array([np.eye(n) + A @ A.T] * 3)
    rc = np.array([1e-6, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    hasil = {m: [] for m in (45, 300)}
    for m in hasil:
        nilai = np.zeros(len(rc))
        for ul in range(20):
            Xl, _ = bangkitkan(m, np.full(3, 1 / 3), mu_b, S_b, rng)
            Xu, _ = bangkitkan(2000, np.full(3, 1 / 3), mu_b, S_b, rng)
            nilai += [GaussianMixture(3, reg_covar=r, n_init=3,
                                      random_state=ul).fit(Xl).score(Xu)
                      for r in rc]
        hasil[m] = nilai / 20
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    for m, w in [(45, BIRU), (300, JINGGA)]:
        ax.semilogx(rc, hasil[m], marker="o", ms=2.5, color=w, lw=0.8,
                    label=f"$m = {m}$")
    ax.set_xlabel("reg\\_covar".replace("\\", ""))
    ax.set_ylabel("log-kem. per sampel uji")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-reg")


# ============================ Bab 11 =================================

def bab11_lunak():
    import warnings
    from sklearn.mixture import GaussianMixture
    from bab01_data import bangkitkan
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH)
    S = np.array([[1.0, 0.3], [0.3, 1.0]])
    Xs, _ = bangkitkan(300, np.array([.5, .5]),
                       np.array([[0., 0.], [2.5, 1.0]]), np.array([S, S]),
                       rng)
    gm = GaussianMixture(2, random_state=0).fit(Xs)
    R = gm.predict_proba(Xs)
    ragu = 1 - R.max(1)
    warna = [tuple(np.array(matplotlib.colors.to_rgb(BIRU)) * r +
                   np.array(matplotlib.colors.to_rgb(JINGGA)) * (1 - r))
             for r in R[:, 0]]
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.2))
    ax[0].scatter(Xs[:, 0], Xs[:, 1], c=warna, s=3)
    ax[0].set_title("warna = $r_{i1}$", fontsize=7)
    sc = ax[1].scatter(Xs[:, 0], Xs[:, 1], c=ragu, s=3, cmap="Reds",
                       vmin=0, vmax=0.5)
    ax[1].set_title("ketidakpastian $1 - \\max_k r_{ik}$", fontsize=7)
    fig.colorbar(sc, ax=ax[1], shrink=0.8).ax.tick_params(labelsize=5.5)
    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("$x_1$")
        a.set_ylabel("$x_2$")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab11-lunak")


def bab11_anomali():
    from bab01_data import akhir_mini, log_normal, logsumexp, log_gabung
    X, g = _mini()
    mu1, S1 = X.mean(0), np.cov(X.T, bias=True)
    a = np.linspace(-1.5, 9.5, 200)
    G = np.array(np.meshgrid(a, a)).reshape(2, -1).T
    s1 = -log_normal(G, mu1, S1).reshape(len(a), len(a))
    s2 = -logsumexp(log_gabung(G, *akhir_mini())).reshape(len(a), len(a))
    batas = (4.4858, 4.0803)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.4))
    for a_, s, b, j in [(ax[0], s1, batas[0], "satu Gaussian"),
                        (ax[1], s2, batas[1], "GMM akhir")]:
        a_.contourf(a, a, s <= b, levels=[0.5, 1.5], colors=[BIRU_MUDA])
        a_.contour(a, a, s, levels=[b], colors=BIRU, linewidths=0.8)
        _titik(a_, X, g, label=False)
        for z, mk in [((4, 4), "x"), ((8, 0), "x")]:
            a_.plot(*z, mk, color=MERAH, ms=5)
        a_.set_title(j, fontsize=7)
        _sumbu_mini(a_)
    fig.tight_layout()
    simpan(fig, "bab11-anomali")


def bab11_klasifikasi():
    import warnings
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.mixture import GaussianMixture
    from bab01_data import bangkitkan
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH + 3)
    s = np.diag([0.5, 1.0])
    X0, _ = bangkitkan(200, np.full(2, .5), np.array([[0., 0.], [4., 0.]]),
                       np.array([s, s]), rng)
    X1, _ = bangkitkan(200, np.full(2, .5), np.array([[2., 0.], [6., 0.]]),
                       np.array([s, s]), rng)
    Xl = np.vstack([X0, X1])
    yl = np.r_[np.zeros(200), np.ones(200)]
    a = np.linspace(-2.5, 8.5, 220)
    b = np.linspace(-4, 4, 160)
    G = np.array(np.meshgrid(a, b)).reshape(2, -1).T
    qda = QuadraticDiscriminantAnalysis().fit(Xl, yl)
    gm = [GaussianMixture(2, random_state=0).fit(Xl[yl == c]) for c in (0, 1)]
    p_gmm = np.column_stack([gm[c].score_samples(G) for c in (0, 1)]).argmax(1)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.9))
    for a_, z, j in [(ax[0], qda.predict(G), "QDA"),
                     (ax[1], p_gmm, "GMM $K = 2$ per kelas")]:
        a_.contourf(a, b, z.reshape(len(b), len(a)), levels=[-0.5, 0.5, 1.5],
                    colors=[BIRU_MUDA, JINGGA_MUDA])
        a_.scatter(X0[:, 0], X0[:, 1], s=1.5, color=BIRU)
        a_.scatter(X1[:, 0], X1[:, 1], s=1.5, color=JINGGA)
        a_.set_title(j, fontsize=7)
        a_.set_xlabel("$x_1$")
        _rapikan(a_)
    fig.tight_layout()
    simpan(fig, "bab11-klasifikasi")


# ============================ Bab 12 =================================

def bab12_keras():
    from bab01_data import bangkitkan, em
    from bab12_variasi import em_keras
    rng = np.random.default_rng(BENIH)
    mu_b = np.array([[0.0, 0.0], [2.0, 0.0]])
    S_b = np.array([np.eye(2)] * 2)
    Xb, z = bangkitkan(4000, np.array([.5, .5]), mu_b, S_b, rng)
    awal = (np.array([.5, .5]), mu_b.copy(), S_b.copy())
    _, mu_l, S_l, _ = em(Xb, *awal, iterasi=500, tol=1e-9)
    _, mu_k, S_k = em_keras(Xb, *awal, iterasi=200)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.2))
    for a, mu, S, j in [(ax[0], mu_l, S_l, "EM lunak"),
                        (ax[1], mu_k, S_k, "EM keras")]:
        a.scatter(Xb[::4, 0], Xb[::4, 1], s=0.6, color=ABU_GARIS)
        for k, w in [(0, BIRU), (1, JINGGA)]:
            _elips(a, mu_b[k], S_b[k], ABU, c=(2,), ls="--")
            _elips(a, mu[k], S[k], w, c=(2,))
        a.set_title(j, fontsize=7)
        a.set_aspect("equal")
        a.set_xlim(-4, 6)
        a.set_ylim(-4, 4)
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab12-keras")


def bab12_t():
    import warnings
    from sklearn.mixture import GaussianMixture
    from bab01_data import bangkitkan
    from bab12_variasi import em_t
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH + 5)
    Xo, _ = bangkitkan(300, np.array([.5, .5]),
                       np.array([[0., 0.], [5., 0.]]),
                       np.array([np.eye(2)] * 2), rng)
    Xo = np.vstack([Xo, rng.uniform([-30, -30], [30, 30], size=(15, 2))])
    awal = np.array([[0.5, 0.5], [4.5, -0.5]])
    mu_t, S_t, _, u = em_t(Xo, awal, np.array([np.eye(2) * 3] * 2),
                           np.array([.5, .5]))
    gm = GaussianMixture(2, means_init=awal, random_state=0).fit(Xo)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.3))
    for a, mu, S, j in [(ax[0], gm.means_, gm.covariances_, "GMM"),
                        (ax[1], mu_t, S_t, "campuran $t$, $\\nu = 4$")]:
        a.scatter(Xo[:300, 0], Xo[:300, 1], s=1, color=ABU)
        a.scatter(Xo[300:, 0], Xo[300:, 1], s=6, marker="x", color=MERAH,
                  lw=0.6)
        for k, w in [(0, BIRU), (1, JINGGA)]:
            _elips(a, mu[k], S[k], w, c=(2,))
        a.set_xlim(-32, 32)
        a.set_ylim(-32, 32)
        a.set_aspect("equal")
        a.set_title(j, fontsize=7)
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab12-t")


def bab12_vb():
    import warnings
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
    from bab01_data import bangkitkan
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH + 7)
    pi_b = np.array([0.5, 0.3, 0.2])
    mu3 = np.array([[0.0, 0.0], [5.0, 1.0], [2.0, 5.0]])
    S3 = np.array([[[2.0, 0.8], [0.8, 1.0]], [[1.0, -0.6], [-0.6, 1.5]],
                   [[0.5, 0.0], [0.0, 2.0]]])
    Xs, _ = bangkitkan(2000, pi_b, mu3, S3, rng)
    w_em = np.sort(GaussianMixture(10, random_state=0).fit(Xs).weights_)[::-1]
    w_vb = np.sort(BayesianGaussianMixture(
        n_components=10, weight_concentration_prior=0.01, max_iter=2000,
        random_state=0).fit(Xs).weights_)[::-1]
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    k = np.arange(1, 11)
    ax.bar(k - 0.2, w_em, width=0.4, color=ABU_GARIS, label="EM, $K = 10$")
    ax.bar(k + 0.2, w_vb, width=0.4, color=BIRU, label="Bayes variasional")
    ax.set_xticks(k)
    ax.set_xlabel("komponen (terurut menurut bobot)")
    ax.set_ylabel("$\\pi_k$")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-vb")


# ============================ Bab 13 =================================

def bab13_graf():
    fig, ax = plt.subplots(figsize=(4.7, 1.6))
    ax.set_xlim(0.3, 6.7)
    ax.set_ylim(0.2, 2.4)
    ax.axis("off")
    for t in range(1, 7):
        ax.add_patch(matplotlib.patches.Circle((t, 1.85), 0.22, fc=BIRU_MUDA,
                                               ec=BIRU, lw=0.7))
        ax.text(t, 1.85, f"$z_{t}$", ha="center", va="center", fontsize=7,
                color=BIRU)
        ax.add_patch(matplotlib.patches.Circle((t, 0.6), 0.22, fc=ABU_GARIS,
                                               ec=ABU, lw=0.7))
        ax.text(t, 0.6, f"$\\mathbf{{x}}_{t}$", ha="center", va="center",
                fontsize=7, color=ABU)
        ax.annotate("", xy=(t, 0.85), xytext=(t, 1.6),
                    arrowprops=dict(arrowstyle="->", color=ABU, lw=0.6))
        if t < 6:
            ax.annotate("", xy=(t + 0.75, 1.85), xytext=(t + 0.25, 1.85),
                        arrowprops=dict(arrowstyle="->", color=BIRU, lw=0.6))
    ax.text(1.5, 2.2, "$A_{jk}$", fontsize=6.5, color=BIRU, ha="center")
    ax.text(1.25, 1.2, "$\\mathcal{N}(\\mu_k, \\Sigma_k)$", fontsize=6,
            color=ABU)
    simpan(fig, "bab13-graf")


def bab13_gamma():
    from bab01_data import awal_mini, data_mini, tanggung_jawab
    from bab13_hmm import emisi, maju_mundur
    X, _ = data_mini()
    pi, mu0, S0 = awal_mini()
    r = tanggung_jawab(X, pi, mu0, S0)[0][:, 0]
    gam = maju_mundur(emisi(X, mu0, S0), pi,
                      np.array([[.75, .25], [.25, .75]]))[0][:, 0]
    t = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(4.7, 1.8))
    ax.bar(t - 0.18, r, width=0.36, color=ABU_GARIS, label="GMM: $r_{t1}$")
    ax.bar(t + 0.18, gam, width=0.36, color=BIRU,
           label="HMM: $\\gamma_t(1)$")
    ax.axhline(0.5, color=ABU, lw=0.5, ls=":")
    ax.set_xticks(t)
    ax.set_xlabel("waktu $t$ (sampel data mini)")
    ax.set_ylabel("peluang keadaan 1")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-gamma")


def bab13_simulasi():
    import warnings
    from sklearn.mixture import GaussianMixture
    from bab13_hmm import emisi, maju_mundur, baum_welch
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH)
    A_b = np.array([[0.95, 0.05], [0.10, 0.90]])
    mu_b = np.array([[0.0, 0.0], [1.5, 1.0]])
    T = 600
    z = np.zeros(T, int)
    for t in range(1, T):
        z[t] = rng.choice(2, p=A_b[z[t - 1]])
    Xs = np.array([rng.multivariate_normal(mu_b[k], np.eye(2)) for k in z])
    gm = GaussianMixture(2, random_state=0).fit(Xs)
    rg = gm.predict_proba(Xs)
    pi, A, mu, S, _ = baum_welch(Xs, np.array([.5, .5]),
                                 np.array([[.8, .2], [.2, .8]]),
                                 gm.means_, gm.covariances_, iterasi=200)
    gam = maju_mundur(emisi(Xs, mu, S), pi, A)[0]
    k1 = int(np.argmax(mu[:, 0]))
    kg = int(np.argmax(gm.means_[:, 0]))
    w = slice(0, 200)
    t = np.arange(T)[w]
    fig, ax = plt.subplots(figsize=(4.7, 2.1))
    ax.fill_between(t, 0, z[w], step="mid", color=JINGGA_MUDA, lw=0,
                    label="keadaan sebenarnya")
    ax.plot(t, rg[w, kg], color=ABU, lw=0.5, label="GMM $r_{t}$")
    ax.plot(t, gam[w, k1], color=BIRU, lw=0.9, label="HMM $\\gamma_t$")
    ax.set_xlabel("waktu $t$")
    ax.set_ylabel("peluang keadaan 2")
    ax.legend(fontsize=5.5, loc="upper right", ncol=3)
    ax.set_ylim(0, 1.25)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-simulasi")


# ============================ Bab 14 =================================

def bab14_info():
    u = np.linspace(-12, 18, 6001)
    du = u[1] - u[0]
    sep = np.linspace(0, 6, 61)
    hilang = []
    for d in sep:
        f1 = np.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi)
        f2 = np.exp(-(u - d) ** 2 / 2) / np.sqrt(2 * np.pi)
        f = 0.5 * f1 + 0.5 * f2
        iobs = np.sum((f1 - f2) ** 2 / f) * du
        hilang.append(1 - iobs / 4)
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    ax.plot(sep, hilang, color=BIRU, lw=0.9)
    ax.set_xlabel("jarak pusat $\\Delta$ (simpangan baku)")
    ax.set_ylabel("pecahan informasi hilang")
    ax.set_ylim(0, 1.02)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-info")


def bab14_label():
    import warnings
    from sklearn.mixture import GaussianMixture
    from bab01_data import bangkitkan
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH)
    pi_b = np.array([0.3, 0.7])
    mu_b = np.array([[0.0, 0.0], [3.0, 1.0]])
    S_b = np.array([[[1.0, 0.3], [0.3, 1.0]], [[1.5, -0.4], [-0.4, 1.0]]])
    mentah, selaras = [], []
    for b in range(300):
        Xb, _ = bangkitkan(200, pi_b, mu_b, S_b, rng)
        gm = GaussianMixture(2, n_init=1, init_params="random_from_data",
                             random_state=b).fit(Xb)
        mentah.append(gm.means_[0, 0])
        selaras.append(gm.means_[np.argmin(gm.means_[:, 0]), 0])
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.9), sharey=True)
    for a, v, j in [(ax[0], mentah, "label mentah"),
                    (ax[1], selaras, "label diselaraskan")]:
        a.hist(v, bins=np.linspace(-1.5, 4, 45), color=BIRU_MUDA,
               edgecolor=ABU_GARIS, lw=0.3)
        a.axvline(0, color=MERAH, lw=0.6, ls="--")
        a.set_title(j, fontsize=7)
        a.set_xlabel("$\\hat\\mu_{11}$")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab14-label")


def bab14_se():
    nama = ["$\\pi_1$", "$\\mu_{11}$", "$\\mu_{12}$", "$\\mu_{21}$",
            "$\\mu_{22}$"]
    import io
    import contextlib
    import runpy
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(Path(__file__).parent / "kode" /
                           "bab14_inferensi.py"), run_name="__main__")
    baris = [b.split() for b in buf.getvalue().splitlines()
             if b.strip().startswith(("pi_1 ", "mu_"))][:5]
    v = np.array([[float(x) for x in b[2:5]] for b in baris])
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    x = np.arange(5)
    for j, (w, t) in enumerate([(ABU_GARIS, "skor (OPG)"),
                                (BIRU, "bootstrap parametrik"),
                                (JINGGA, "Monte Carlo")]):
        ax.bar(x + (j - 1) * 0.26, v[:, j], width=0.26, color=w, label=t)
    ax.set_xticks(x)
    ax.set_xticklabels(nama)
    kunci_label(ax, "x")
    ax.set_ylabel("galat baku")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-se")


# ============================ Bab 15 =================================

_WARNA_SP = {"Adelie": BIRU, "Chinstrap": JINGGA, "Gentoo": HIJAU}


def bab15_data():
    from bab15_studikasus import muat
    d, X, Z, _, _ = muat()
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.3))
    for sp, w in _WARNA_SP.items():
        s = (d.species == sp).to_numpy()
        ax[0].scatter(X[s, 0], X[s, 1], s=2, color=w, label=sp)
        ax[1].scatter(X[s, 2], X[s, 3] / 1000, s=2, color=w)
    ax[0].set_xlabel("panjang paruh (mm)")
    ax[0].set_ylabel("kedalaman paruh (mm)")
    ax[1].set_xlabel("panjang sirip (mm)")
    ax[1].set_ylabel("massa (kg)")
    ax[0].legend(fontsize=5.5, markerscale=3, loc="upper right")
    for a in ax:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab15-data")


def bab15_bic():
    import warnings
    from sklearn.mixture import GaussianMixture
    from bab15_studikasus import muat, icl
    warnings.filterwarnings("ignore")
    _, _, Z, _, _ = muat()
    Ks = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(4.7, 2.2))
    for j, w, mk in [("full", BIRU, "o"), ("tied", JINGGA, "s"),
                     ("diag", HIJAU, "^"), ("spherical", ABU, "d")]:
        b, c = [], []
        for K in Ks:
            gm = GaussianMixture(K, covariance_type=j, n_init=10,
                                 random_state=0).fit(Z)
            b.append(gm.bic(Z))
            c.append(icl(gm, Z))
        ax.plot(Ks, b, marker=mk, ms=2.5, color=w, lw=0.8, label=f"BIC {j}")
        if j == "tied":
            ax.plot(Ks, c, marker=mk, ms=2.5, color=w, lw=0.8, ls="--",
                    label="ICL tied")
    ax.set_ylim(2450, 3100)
    ax.set_xlabel("banyaknya komponen $K$")
    ax.set_ylabel("nilai kriteria")
    ax.legend(fontsize=5.5, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab15-bic")


def bab15_klaster():
    import warnings
    from sklearn.mixture import GaussianMixture
    from bab15_studikasus import muat
    warnings.filterwarnings("ignore")
    d, X, Z, rata, sb = muat()
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.3))
    warna = [BIRU, JINGGA, HIJAU, MERAH]
    for a, K in zip(ax, (3, 4)):
        gm = GaussianMixture(K, covariance_type="tied", n_init=10,
                             random_state=0).fit(Z)
        lab = gm.predict(Z)
        o = np.argsort(np.argsort(gm.means_[:, 2]))
        for k in range(K):
            s = lab == k
            a.scatter(X[s, 0], X[s, 1], s=2, color=warna[o[k]])
            mu = gm.means_[k][:2] * sb[:2] + rata[:2]
            S = gm.covariances_[:2, :2] * np.outer(sb[:2], sb[:2])
            _elips(a, mu, S, warna[o[k]], c=(2,))
        a.set_title(f"tied, $K = {K}$", fontsize=7)
        a.set_xlabel("panjang paruh (mm)")
        a.set_ylabel("kedalaman paruh (mm)")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab15-klaster")


# Fungsi gambar baru disisipkan DI ATAS penanda ini.



if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
