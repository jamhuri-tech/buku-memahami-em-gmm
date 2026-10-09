# Kode — Memahami EM dan Gaussian Mixture

Repositori pendamping buku **_Memahami EM dan Gaussian Mixture:
Variabel Laten dan Klasterisasi Lunak, Iterasi demi Iterasi_** (Edisi
Pertama, 2026) oleh Mohammad Jamhuri, seri *Memahami*. Tag `edisi-1`
akan menandai kode yang tepat dipakai untuk mencetak edisi pertama.

Buku ini memakai satu **data mini** dari awal sampai akhir: enam
pelanggan dengan dua fitur, (1, 1), (1, 4), (4, 1), (4, 7), (7, 4),
(7, 7), dan kelompok pembanding g = (1, 1, 1, 2, 2, 2) di kolom
terakhir. Dari pemulaan μ1 = (3, 3), μ2 = (5, 5), dan σ0² = 6/log 3,
tanggung jawab iterasi pertama EM tepat (9/10, 3/4, 3/4, 1/4, 1/4,
1/10), sehingga satu iterasi penuh dapat dihitung dengan pecahan.
Setiap tanggung jawab, langkah M, batas ELBO, laju konvergensi, BIC,
dan galat baku dihitung tangan di buku lalu diperiksa oleh kode di
sini. File `kode/babNN_contoh.py` memeriksa setiap bilangan di kotak
Contoh Soal Bab NN.

**Setiap angka keluaran yang tercetak di buku dihasilkan oleh kode di
sini**, dan `periksa.py` membuktikannya: skrip itu menjalankan ulang
kode setiap bab dan mencocokkan hasilnya dengan blok keluaran yang
tercetak di buku.

Seluruh kode boleh dipakai, disalin, diubah, dan disebarluaskan secara
bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban
mencantumkan sumber (lisensi [0BSD](LICENSE)). Data Palmer Penguins di
`data/penguins/` dirilis dengan lisensi CC0 1.0 (`data/penguins/LICENSE.md`);
asal dan rujukannya ada di `data/penguins/SUMBER.md`.

## Menjalankan

```bash
git clone https://github.com/jamhuri-tech/buku-memahami-em-gmm.git
cd buku-memahami-em-gmm
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python kode/bab01_campuran.py
```

Setiap skrip dijalankan dari akar repositori. Pembangkit bilangan acak
selalu memakai benih tetap (20261009), sehingga keluarannya sama setiap
kali dijalankan.

## Memeriksa angka di buku

```bash
.venv/bin/python periksa.py        # semua bab
.venv/bin/python periksa.py 04     # Bab 4 saja
```

Keluaran `SEMUA COCOK` berarti setiap blok keluaran di buku dihasilkan
ulang oleh kode ini. Pemeriksaan yang sama berjalan otomatis di GitHub
Actions setiap kali kode berubah.

## Struktur

```
kode/
  bab01_data.py      data mini dan fungsi dasar GMM (log-kepadatan,
                     tanggung jawab, langkah M, EM)
  babNN_*.py         kode Bab NN
  babNN_contoh.py    pemeriksa hitungan tangan Contoh Soal Bab NN
  lampiran_gmm.py    EM untuk GMM dengan NumPy saja (Lampiran B)
data/penguins/       data Palmer Penguins (Bab 15), lisensi, sumber
keluaran/babNN.txt   blok keluaran yang tercetak di Bab NN
gen_gambar.py        membangkitkan semua gambar buku ke gbr/
periksa.py           mencocokkan kode dengan keluaran/
requirements.txt     versi pustaka yang dipakai buku
```

`python gen_gambar.py bab04` membangkitkan gambar Bab 4 saja.

## Versi

`requirements.txt` mematok versi yang dipakai buku (Python 3.13,
NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, pandas 2.2.3,
Matplotlib 3.10.0). Versi lain biasanya berjalan, tetapi digit terakhir
sebagian angka dapat berbeda.

## Salah ketik, galat, dan saran

Silakan buka [issue](../../issues) di repositori ini: sebutkan bab,
halaman, dan apa yang keliru. Saran juga dapat dikirim ke
m.jamhuri@live.com.
