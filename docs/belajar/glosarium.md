# Glosarium

Istilah kunci yang dipakai di catatan belajar. Bertumbuh tiap bab (Fase 1).

| Istilah | Arti singkat |
|---|---|
| **Token** | Potongan teks (bisa kata, sub-kata, atau karakter) yang menjadi satuan masukan/keluaran LLM. |
| **Kosakata (vocabulary)** | Himpunan semua token yang dikenal model; ukurannya `V`. |
| **`<bos>` / `<eos>`** | Token buatan penanda awal (`beginning of sequence`) dan akhir (`end of sequence`) urutan. |
| **Logit** | Angka mentah keluaran model sebelum diubah menjadi probabilitas (Bab 10). |
| **Softmax** | Fungsi yang mengubah deretan logit menjadi probabilitas (0–1, jumlah 1). Bab 10. |
| **Log-sum-exp (LSE)** | `ln ∑ e^{z_j}`; muncul saat menyatukan softmax + cross-entropy: `L = LSE(z) − z_c`. Versi stabil: kurangi `max` dulu. |
| **Gradien `p − y`** | Gradien loss (softmax + cross-entropy) terhadap logit = prediksi − jawaban; besarnya otomatis sebanding kesalahan. Bab 10. |
| **Label smoothing** | Melunakkan label one-hot: `(1−ε)y + ε/V`; regularisasi agar logit tidak tumbuh tanpa batas. Bab 10. |
| **MSE (mean squared error)** | Loss `∑(p−y)²`; gradiennya lenyap saat model salah besar → ditinggalkan untuk prediksi token. Bab 10. |
| **Temperature** | Pengali skala logit sebelum softmax; mengatur keruncingan distribusi. Bab 19. |
| **Parameter** | Angka model yang ditentukan lewat pelatihan lalu dibekukan. |
| **Autoregressive generation** | Menghasilkan teks dengan mengulang prediksi satu token, menempelkannya ke masukan, lalu mengulang. |
| **KV cache** | Penyimpanan vektor key/value token lama agar inferensi tidak menghitung ulang semuanya. Bab 19. |
| **Distribusi probabilitas** | Daftar angka non-negatif yang jumlahnya 1, menyatakan kemungkinan tiap token. |
| **Skalar** | Satu angka tunggal (ditulis huruf kecil biasa, mis. `x`). |
| **Vektor** | Deretan angka berurutan (ditulis huruf kecil tebal, mis. **x**); simbol `‖x‖` = panjangnya. |
| **Matriks** | Tabel angka baris × kolom (ditulis huruf besar tebal, mis. **X**); `X_ij` = elemen baris `i` kolom `j`. |
| **Indeks** | Nomor urut elemen (angka kecil di bawah, mis. `x_3`). Buku memakai indeks mulai 1; kode mulai 0. |
| **Eksponensial `eˣ`** | Fungsi yang selalu bernilai positif dan naik cepat; `e ≈ 2,71828`. |
| **Logaritma natural `ln`** | Kebalikan `eˣ`; mengubah perkalian menjadi penjumlahan. Basis `e` (nat); basis 2 untuk bit. |
| **Nat / bit** | Satuan informasi: 1 nat = logaritma basis `e`, 1 bit = logaritma basis 2. Bab 6. |
| **argmax** | Indeks (posisi) tempat nilai terbesar berada — dipakai memilih token, bukan nilainya. |
| **Dot product** | Operasi dua vektor → satu skalar: `∑ a_i b_i`; ukuran kecocokan arah & besar. |
| **Norma (L2)** | Panjang vektor: `√(∑ x_i²)`; dasar RMSNorm & LayerNorm. |
| **Cosine similarity** | Dot product dibagi panjang kedua vektor; hanya mengukur kemiripan arah, rentang `[−1, +1]`. |
| **Hadamard product** | Perkalian elemen-per-elemen dua vektor (`a ⊙ b`); hasilnya vektor. |
| **d_model** | Dimensi vektor tiap token (mis. 768 untuk GPT-2 small, 12288 untuk GPT-3). |
| **Transpos (Aᵀ)** | Matriks A dengan baris dan kolom ditukar; dipakai di `QKᵀ`. |
| **Transformasi linear** | Operasi `y = xW`; menumpuknya tidak menambah kemampuan (butuh non-linear). |
| **FLOP** | Satu operasi aritmetika pada bilangan desimal; ukuran biaya komputasi. |
| **Batch** | Memproses banyak token sekaligus lewat satu perkalian matriks. |
| **Ruang sampel (Ω)** | Himpunan semua hasil yang mungkin; untuk LLM = kosakata. |
| **Probabilitas bersyarat** | `P(A\|B) = P(A ∩ B)/P(B)`; model bahasa = kalkulator probabilitas bersyarat. |
| **Ekspektasi `E[·]`** | Rata-rata berbobot: `∑ P(x) f(x)`. |
| **Varians / simpangan baku** | Sebaran nilai di sekitar rata-rata; dasar normalisasi LayerNorm. |
| **Distribusi normal** | Sebaran Gaussian `N(μ, σ²)`; untuk inisialisasi parameter & analisis skala. |
| **Sampling** | Memilih satu token dari distribusi (inverse transform sampling). |
| **Entropi** | Rata-rata kejutan sebuah distribusi `H(P) = −∑ P log P`. |
| **Cross-entropy** | `H(P,Q) = −∑ P log Q`; fungsi loss LLM. |
| **KL divergence** | Selisih dua distribusi `D_KL(P‖Q) = H(P,Q) − H(P) ≥ 0`; tak simetris. |
| **Perplexity** | `e^L`; jumlah pilihan efektif yang masih dipertimbangkan model. |
| **BPB/BPC** | Bits per byte/character; loss ternormalisasi agar bisa dibandingkan antar tokenizer. |
| **Turunan** | Laju perubahan keluaran terhadap masukan; `lim (f(x+h) − f(x))/h`. |
| **Turunan parsial (∂)** | Turunan terhadap satu variabel dengan variabel lain dianggap konstan. |
| **Gradien (∇)** | Vektor semua turunan parsial; menunjuk arah naik paling cepat. |
| **Learning rate (η)** | Ukuran langkah gradient descent; terlalu besar → loss meledak. |
| **Gradient descent** | Menurunkan loss dengan menggeser parameter ke arah `−∇L`. |
| **SGD (stochastic GD)** | Gradient descent dengan gradien dari mini-batch; derau ∝ `1/√B`. Bab 12. |
| **Mini-batch** | Sebagian kecil data (32–jutaan token) untuk satu langkah pembaruan. Bab 12. |
| **Momentum** | Mengakumulasi rata-rata bergerak gradien (`v ← βv+g`) → arah konsisten dipercepat. Bab 12. |
| **Adam** | Optimizer dengan learning rate per parameter (`m̂/√v̂`); kebal penskalaan gradien. Bab 12. |
| **Momen pertama / kedua** | Rata-rata gradien (`m`) dan rata-rata kuadrat gradien (`v`) di Adam. Bab 12. |
| **Koreksi bias** | Membagi `m,v` dengan `1−β^t` agar tak terlalu kecil di langkah awal. Bab 12. |
| **AdamW** | Adam dengan weight decay terpisah (`+λθ`), standar untuk LLM. Bab 12. |
| **Weight decay** | Regularisasi mendorong parameter ke nol; λ umum 0,1. Bab 12. |
| **Warmup** | Menaikkan learning rate secara linear di awal pelatihan. Bab 12. |
| **Cosine decay** | Menurunkan learning rate mengikuti kurva kosinus sampai `η_min`. Bab 12. |
| **Gradient clipping** | Memotong norma gradien ke batas `c` (`g ← g·c/‖g‖`) agar tidak meledak. Bab 12. |
| **Attention** | Mekanisme memindahkan informasi antarposisi: rata-rata berbobot dari value. Bab 13. |
| **Query / Key / Value** | Tiga cara memandang token yang sama (`H·W_Q`, `H·W_K`, `H·W_V`); cari / ditawarkan / diambil. Bab 13. |
| **Scaled dot-product attention** | `softmax(QKᵀ/√d_k + M)·V`. Bab 13. |
| **`√d_k` (skala)** | Menjaga varians skor `q·k` ≈ 1 agar softmax tidak runtuh jadi one-hot. Bab 13. |
| **Causal mask** | Menutup posisi masa depan (`−∞`) agar model tidak mencontek. Bab 13. |
| **Multi-head attention** | Beberapa attention paralel di subruang `d/h`, digabung lewat `W_O`. Bab 14. |
| **W_O (proyeksi keluaran)** | Matriks penggabung antar-head; satu-satunya tempat hasil head bercampur. Bab 14. |
| **KV cache** | Simpanan K/V token sebelumnya saat inference; `2·L·h·n·d_k·byte`. Bab 14/19. |
| **MQA / GQA** | Berbagi K/V antar-head untuk mengecilkan KV cache (`g` grup untuk GQA). Bab 14. |
| **Induction head** | Head yang menemukan kemunculan sebelumnya suatu token lalu menyalin yang mengikutinya (temuan empiris). Bab 14. |
| **Informasi posisi** | Cara memberitahu model urutan token; attention sendiri buta urutan. Bab 15. |
| **Positional embedding (learned)** | Tabel posisi yang dilatih, **dijumlahkan** ke embedding: `H(0) = X + P`. Bab 15. |
| **Sinusoidal encoding** | Posisi dihitung dengan `sin`/`cos` pada berbagai frekuensi; tanpa parameter. Bab 15. |
| **RoPE** | Memutar query/key sebesar sudut ∝ posisi; `q'_m·k'_n` hanya bergantung `m − n`. Bab 15. |
| **ALiBi** | Menambahkan penalti linear jarak langsung ke skor attention. Bab 15. |
| **Equivariance (permutasi)** | Attention tanpa mask: mengacak baris masukan hanya mengacak keluaran dengan cara sama, isi tetap. Bab 15. |
| **Position interpolation** | Menskalakan posisi (mis. 8192 → 2048) untuk memperpanjang konteks RoPE. Bab 15. |
| **Blok Transformer** | Dua sub-blok berbalut residual: attention lalu FFN; masuk `n×d`, keluar `n×d`. Bab 16. |
| **Pre-LN / Post-LN** | Letak normalisasi: sebelum sub-blok (Pre-LN, modern) atau setelah residual (Post-LN, mini-LLM). Bab 16. |
| **LayerNorm** | Normalisasi per token sepanjang fitur: `γ(h−μ)/√(σ²+ε)+β`. Bab 16. |
| **RMSNorm** | LayerNorm tanpa mean & β: `γ·h/√(mean(h²)+ε)`. Bab 16. |
| **Lapisan keluaran (unembedding)** | Mengubah `n×d` menjadi logit `n×V`; dengan weight tying `Z = H·Eᵀ`. Bab 17. |
| **Weight tying** | Memakai matriks embedding (ditranspos) sebagai `W_out`, hemat `V×d` parameter. Bab 8/17. |
| **Keadaan akhir (final state)** | Vektor `d` dimensi hasil blok terakhir untuk satu posisi; memuat seluruh konteks. Bab 17. |
| **Pretraining** | Melatih model menebak token berikutnya pada teks mentah. Bab 18. |
| **Self-supervised** | Label diambil dari data itu sendiri (token berikutnya), bukan anotasi manusia. Bab 18. |
| **Gradient accumulation** | Menjumlah gradien `k` sub-batch lalu memperbarui sekali; hemat memori. Bab 18. |
| **Scaling law** | Loss turun mengikuti hukum pangkat `L(N) ∝ N^{−α}`, α ≈ 0,076. Bab 18. |
| **Chinchilla** | Alokasi optimal `D ≈ 20·N` token per parameter. Bab 18. |
| **Perplexity** | `e^L`; jumlah pilihan efektif yang masih dipertimbangkan model. Bab 6/21. |
| **Greedy decoding** | Memilih token berprobabilitas tertinggi (`argmax`). Bab 19. |
| **Temperature** | Pengali skala logit sebelum softmax; mengatur keruncingan distribusi. Bab 19. |
| **Top-k / top-p** | Membatasi kandidat sampling: `k` teratas (tetap) / sampai kumulatif `p` (adaptif). Bab 19. |
| **Beam search** | Melacak `b` urutan terbaik; jarang untuk chat (cenderung membosankan). Bab 19. |
| **Prefill / decode** | Fase inference: proses prompt sekaligus (FLOP) vs hasilkan token satu-satu (bandwidth memori). Bab 19. |
| **Mini-LLM 188 parameter** | Transformer terkecil utuh untuk memverifikasi matematika buku; bukan model pakai. Bab 21. |
| **SFT** | Supervised fine-tuning: lanjutkan pretraining pada instruksi–jawaban, loss hanya di jawaban. Bab 20. |
| **Reward model** | Model yang memberi satu skor pada (prompt, jawaban); dilatih dari perbandingan manusia. Bab 20. |
| **Bradley–Terry** | `P(y_w≻y_l)=σ(r_w−r_l)`; dasar loss reward model & DPO. Bab 20. |
| **RLHF / PPO** | Menyelaraskan model dengan reward + penalti KL; dioptimasi via PPO. Bab 20. |
| **Reward hacking** | Model mengeksploitasi celah reward model; dicegah suku KL. Bab 20. |
| **DPO** | Menyelaraskan langsung dari pasangan preferensi (tanpa reward model/RL). Bab 20. |
| **LoRA** | Fine-tuning hemat: `W' = W + (α/r)BA`; latih `A,B` saja. Bab 20. |
| **Port PyTorch** | Implementasi kedua yang harus setara dengan NumPy (uji paritas, ADR-0001). Bab 20/Fase 11. |
| **Aturan rantai** | `dz/dx = (dz/dy)(dy/dx)`; dasar backpropagation. |
| **Backpropagation** | Gradien merambat mundur dari loss, mengalikan turunan lokal. Biaya ≈ 2× forward. |
| **Error signal (δ)** | `∂L/∂z^(ℓ)`, delta pada lapisan `ℓ`; besaran yang dirambatkan mundur. Bab 11. |
| **Residual connection** | Struktur `y = x + F(x)`; menyediakan jalur identitas agar gradien lewat tanpa dikali. Bab 11. |
| **Jacobian (J_F)** | Matriks semua turunan parsial; `∇_x L = (I + J_F)ᵀ ∇_y L` untuk residual. |
| **Gradient checkpointing** | Menyimpan lebih sedikit aktivasi lalu menghitungnya ulang saat backward — hemat memori. Bab 11. |
| **Rumus biaya latih** | `C ≈ 6·N·D` FLOP (N parameter, D token); angka 6 = 2 FLOP/param/token × 3 (maju + 2 mundur). Bab 11. |
| **Vanishing gradient** | Gradien menyusut hingga nol pada aktivasi jenuh (mis. sigmoid berlapis). |
| **Gradien numerik** | Verifikasi gradien lewat central difference `(L(θ+ε) − L(θ−ε))/(2ε)`. |
| **Tokenisasi** | Memecah teks menjadi token (sub-kata); teks → deretan ID integer. |
| **BPE** | Byte Pair Encoding; menggabungkan pasangan simbol paling sering, berulang. |
| **One-hot** | Vektor panjang `V` bernilai 1 di posisi token, 0 di tempat lain. |
| **Embedding** | Baris matriks `E ∈ ℝ^{V×d}` untuk tiap token; nilainya dipelajari, bukan ditulis. |
| **Weight tying** | Memakai matriks embedding yang sama (ditranspos) untuk lapisan keluaran: `logit = h Eᵀ`. |
| **`<unk>` / `<bos>` / `<eos>`** | Token khusus: tak dikenal / awal urutan / akhir urutan. |
| **Neuron** | Satuan terkecil FFN: `y = f(x · w + b)`; menjumlahkan masukan berbobot lalu melewatkannya ke aktivasi. |
| **Pre-aktivasi (`z`)** | Nilai sebelum aktivasi: `z = x · w + b`. |
| **Lapisan linear (dense)** | Banyak neuron sekaligus: `y = f(xW + b)`; `n·m + m` parameter. Dinamai *fully connected*. |
| **Fungsi aktivasi** | Fungsi non-linear yang dipilih perancang (tidak dilatih); membuat kedalaman model berarti. |
| **ReLU** | `max(0, x)`; murah, gradien 1 atau 0; neuron bisa "mati" (*dying ReLU*). |
| **GELU** | `x·Φ(x)` (Φ = CDF normal standar); aktivasi GPT-2/GPT-3/BERT. |
| **SiLU / Swish** | `x·σ(x)`; mirip GELU; komponen SwiGLU di Llama. |
| **SwiGLU** | FFN bergerbang: `SiLU(xW_gate) ⊙ (xW_up)`; 3 matriks, `d_ff` dikurangi agar parameter setara. |
| **FFN (feed-forward network)** | Dua lapisan di tiap blok Transformer: naik ke `d_ff`, aktivasi, turun ke `d`. Memuat ~2/3 parameter. |
| **d_ff** | Dimensi lapisan feed-forward; konvensi `4d` (SwiGLU memakai `(2/3)·4d`). |
| **Dying ReLU** | Neuron yang berhenti belajar karena pre-aktivasi selalu negatif → gradien nol terus-menerus. |
