"""Bab 15: studi kasus GMM pada Palmer Penguins (empat ukuran tubuh)."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)
BENIH = 20261009
FITUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm",
         "body_mass_g"]


def muat():
    d = pd.read_csv(Path(__file__).parent.parent / "data" / "penguins" /
                    "penguins.csv")
    d = d.dropna(subset=FITUR).reset_index(drop=True)
    X = d[FITUR].to_numpy(float)
    rata, sb = X.mean(0), X.std(0)
    return d, X, (X - rata) / sb, rata, sb


def icl(gm, Z):
    R = gm.predict_proba(Z)
    return gm.bic(Z) - 2 * np.sum(R * np.log(np.clip(R, 1e-300, 1)))


if __name__ == "__main__":
    d, X, Z, rata, sb = muat()
    m = len(Z)

    # (1) Data
    print("(1) Palmer Penguins: m = %d (2 baris tanpa ukuran dibuang)" % m)
    print("    spesies:", d.species.value_counts().to_dict())
    print("    rata-rata:", rata.round(1))
    print("    simp. baku:", sb.round(1))
    print("    korelasi panjang paruh-sirip   = %.3f"
          % np.corrcoef(X[:, 0], X[:, 2])[0, 1])
    print("    korelasi kedalaman paruh-sirip = %.3f"
          % np.corrcoef(X[:, 1], X[:, 2])[0, 1])

    # (2) Kisi BIC dan ICL
    hasil = []
    for j in ("full", "tied", "diag", "spherical"):
        for K in range(1, 9):
            gm = GaussianMixture(K, covariance_type=j, n_init=10,
                                 random_state=0).fit(Z)
            hasil.append((gm.bic(Z), icl(gm, Z), j, K, gm._n_parameters(),
                          gm.score(Z) * m))
    print("(2) Lima BIC terkecil dari 32 model")
    print("    (K = 1..8, empat struktur kovarians):")
    print("    struktur  K   p   log-kem     BIC      ICL")
    for b, c, j, K, p, l in sorted(hasil)[:5]:
        print("    %-9s %d  %2d  %9.2f  %7.1f  %7.1f" % (j, K, p, l, b, c))
    b, c, j, K, p, l = min(hasil, key=lambda r: r[1])
    print("    ICL terkecil: %s K = %d (ICL %.1f, BIC %.1f)" % (j, K, c, b))

    # (3) Dua calon: tied K = 3 dan tied K = 4
    print("(3) Diagnostik dua calon (kovarians bersama):")
    for K in (3, 4):
        gm = GaussianMixture(K, covariance_type="tied", n_init=10,
                             random_state=0).fit(Z)
        lab = gm.predict(Z)
        R = gm.predict_proba(Z)
        terbaik = gm.score(Z) * m
        ll = [GaussianMixture(K, covariance_type="tied", random_state=s)
              .fit(Z).score(Z) * m for s in range(20)]
        print("    K = %d: log-kem = %.2f, konvergen = %s" % (K, terbaik,
                                                           gm.converged_))
        print("      pemulaan tunggal mencapai terbaik: %d dari 20"
              % sum(v > terbaik - 0.5 for v in ll))
        print("      N_k =", np.sort(R.sum(0)).round(1))
        print("      eigen terkecil Sigma = %.3f"
              % np.linalg.eigvalsh(gm.covariances_).min())
        print("      rata-rata ketidakpastian = %.4f, ARI spesies = %.3f"
              % ((1 - R.max(1)).mean(), adjusted_rand_score(d.species, lab)))
        tab = pd.crosstab(d.species, lab)
        tab.columns = ["k%d" % (c + 1) for c in tab.columns]
        print("      " + tab.to_string().replace("\n", "\n      "))
        if K == 4:
            ad = (d.species == "Adelie").to_numpy()
            jk = pd.crosstab(d.sex[ad].fillna("?"), lab[ad])
            jk.columns = ["k%d" % (c + 1) for c in jk.columns]
            print("      Adelie menurut jenis kelamin:")
            print("      " + jk.to_string().replace("\n", "\n      "))
    km = KMeans(3, n_init=10, random_state=0).fit(Z)
    print("    pembanding K-Means K = 3: ARI spesies = %.3f"
          % adjusted_rand_score(d.species, km.labels_))

    # (4) Pusat komponen K = 3 dalam satuan asal, bootstrap galat baku bobot
    gm = GaussianMixture(3, covariance_type="tied", n_init=10,
                         random_state=0).fit(Z)
    o = np.argsort(gm.means_[:, 2])
    pusat = gm.means_[o] * sb + rata
    rng = np.random.default_rng(BENIH)
    L = np.linalg.cholesky(gm.covariances_)
    boot = []
    for b in range(100):
        z = rng.choice(3, size=m, p=gm.weights_)
        Zb = gm.means_[z] + rng.standard_normal((m, 4)) @ L.T
        gb = GaussianMixture(3, covariance_type="tied", n_init=2,
                             random_state=b).fit(Zb)
        boot.append(gb.weights_[np.argsort(gb.means_[:, 2])])
    se = np.array(boot).std(0)
    print("(4) GMM K = 3, pusat dalam satuan asal (urut sirip):")
    print("    paruh-p  paruh-d  sirip   massa    bobot  SE bootstrap")
    for k in range(3):
        print("    %6.1f   %6.1f   %5.1f   %6.0f   %.3f  %.3f"
              % (*pusat[k], gm.weights_[o][k], se[k]))
