# Cheatsheet — Rumus & Simbol

Ringkasan rumus dan simbol yang muncul di catatan belajar. Bertumbuh tiap bab (Fase 1).

## Notasi dasar

| Simbol | Arti |
|---|---|
| `x` | skalar (satu angka) |
| **x** | vektor (deretan angka) |
| **X** | matriks (tabel angka) |
| `x_i` | elemen ke-`i` dari vektor **x** |
| `X_ij` | elemen baris `i`, kolom `j` dari matriks **X** |
| `x_t` | token (atau nilainya) pada posisi ke-`t` |
| `V` | ukuran kosakata (jumlah token) |
| `d` / `d_model` | dimensi model (panjang vektor tiap token) |
| `L` | jumlah lapisan Transformer |
| `d_ff` | dimensi lapisan feed-forward (umumnya `4d`) |
| `∑` (sigma) | penjumlahan semua suku |
| `∏` (pi) | perkalian semua suku |
| `eˣ` / `exp(x)` | eksponensial (selalu positif) |
| `ln(x)` | logaritma natural (kebalikan `eˣ`) |
| `∂` | turunan parsial |
| `∇` | gradien (kumpulan turunan parsial) |
| `‖x‖` | norma (panjang) vektor `x` |
| `x · y` | dot product |
| `xᵀ` | transpos dari `x` |
| `max` / `argmax` | nilai terbesar / posisi nilai terbesar |
| `𝔼[·]` | ekspektasi (nilai rata-rata) |
| `∼` | berdistribusi sebagai |
| `≈` | kira-kira sama dengan |
| `∝` | sebanding dengan (sampai faktor konstan) |
| `∈`, `ℝⁿ` | anggota dari; himpunan vektor `n` bilangan real |

## Rumus per bab

### Bab 1 — Apa yang dihitung LLM
- Prediksi token berikutnya: `P(x_{t+1} | x_1, …, x_t)`
- Probabilitas kalimat (aturan rantai): `P(x_1, …, x_n) = ∏_{t=1}^{n} P(x_t | x_1, …, x_{t-1})`
- Jumlah parameter Transformer decoder: `N ≈ V·d + L·(4d² + 2·d·d_ff)`; jika `d_ff = 4d` maka `N ≈ V·d + L·12d²`

### Bab 2 — Kamus notasi
- Penjumlahan: `∑_{i=1}^{n} a_i = a_1 + a_2 + … + a_n`
- Perkalian: `∏_{i=1}^{n} a_i = a_1 × a_2 × … × a_n`
- Eksponensial: `e^{a+b} = e^a × e^b`; `e⁰ = 1`; `eˣ > 0` selalu
- Logaritma: `ln(1) = 0`; `ln(a × b) = ln(a) + ln(b)`; `ln(x) < 0` untuk `0 < x < 1`
- Ganti basis: `log₂(x) = ln(x) / ln(2)`
- Loss LLM: `−ln(p)` (positif karena `ln(p)` negatif untuk `p ∈ (0,1)`)

### Bab 3 — Vektor
- Penjumlahan: `a + b = [a_1+b_1, …, a_n+b_n]` (dimensi harus sama)
- Kali skalar: `c·x = [c·x_1, …, c·x_n]`
- Dot product: `a · b = ∑_{i=1}^{n} a_i b_i` → dua vektor jadi satu skalar
- Norma L2: `‖x‖ = √(∑ x_i²)`; hubungan `‖x‖² = x · x`
- Cosine similarity: `cos(θ) = (a·b)/(‖a‖‖b‖) ∈ [−1, +1]`
- Hadamard (elemen-per-elemen): `(a ⊙ b)_i = a_i b_i` → hasil vektor
- `Var(q·k) = d_k` untuk komponen normal standar (alasan pembagian `√d_k` di attention)

### Bab 4 — Matriks
- Ukuran: `A ∈ ℝ^{m×n}` (baris dulu, lalu kolom); `A_ij` = baris `i`, kolom `j`
- Transpos: `(Aᵀ)_ij = A_ji`; `(AB)ᵀ = BᵀAᵀ`
- Perkalian: `C_ij = ∑_p A_ip B_pj`; syarat dimensi tengah cocok: `(m×k)·(k×n)=(m×n)`
- Perkalian tidak komutatif: `AB ≠ BA`
- Transformasi linear: `y = xW`; dua lapisan linear = satu lapisan → non-linear wajib (Bab 9)
- Batch: `Q = HW_Q` menghasilkan query untuk semua token sekaligus
- Biaya: `FLOP ≈ 2·m·n·k`; attention berskala `n²`

### Bab 5 — Probabilitas
- Distribusi: `0 ≤ P(ω) ≤ 1` dan `∑ P(ω) = 1`
- Seragam: `P(ω) = 1/|Ω|`
- Bersyarat: `P(A | B) = P(A ∩ B) / P(B)`
- Aturan rantai: `P(x_1,…,x_n) = ∏_{t=1}^{n} P(x_t | x_{<t})` (berlaku persis)
- Log-prob: `log P(x_1,…,x_n) = ∑ log P(x_t | x_{<t})` (stabilitas numerik)
- Ekspektasi: `E[f] = ∑ P(x) f(x)`; varians `Var(x) = E[(x−E[x])²]`; `σ = √Var`
- Normalisasi LayerNorm: `ĥ = (h − μ)/σ`
- Normal: `p(x) = (1/(σ√(2π))) exp(−(x−μ)²/(2σ²))`
- Sampling: inverse-CDF — pilih token pertama yang `C_k > u`

### Bab 6 — Teori informasi
- Informasi: `I(p) = −log p` (basis 2 → bit, `ln` → nat)
- Entropi: `H(P) = −∑ P log P`; maksimum `log n` saat seragam
- Cross-entropy: `H(P,Q) = −∑ P log Q` ≥ `H(P)`; `= H(P)` hanya jika `P = Q`
- KL divergence: `D_KL(P‖Q) = ∑ P log(P/Q) = H(P,Q) − H(P) ≥ 0` (tak simetris)
- Loss LLM (target one-hot): `L = −log P_θ(x_benar)`
- Loss rata-rata: `L = −(1/N) ∑ log P_θ(x_t | x_{<t})`
- Perplexity: `PPL = e^L`; `PPL = k` ⇔ seakan memilih dari `k` pilihan
- Bit per byte: `BPB = L_total(nat) / (ln 2 × jumlah byte)`

### Bab 7 — Turunan, gradien, aturan rantai
- Turunan: `f'(x) = lim_{h→0} (f(x+h) − f(x))/h`
- Aturan update (gradient descent): `θ ← θ − η ∇_θ L`
- Parsial: `∂f/∂x` (variabel lain ditahan); gradien `∇f = [∂f/∂x₁, …]`
- Aturan rantai: `dz/dx = (dz/dy)(dy/dx)` — dasar backpropagation
- Percabangan: gradien dijumlahkan (residual connection)
- Turunan `y = xW`: `∂L/∂W = xᵀ (∂L/∂y)`, `∂L/∂x = (∂L/∂y) Wᵀ`
- Gradien numerik (verifikasi): `∂L/∂θ_i ≈ (L(θ_i+ε) − L(θ_i−ε)) / (2ε)`
- Turunan loss terhadap logit = `p − y` (softmax + cross-entropy)

### Bab 8 — Tokenisasi & embedding
- Tokenisasi sub-kata (BPE): gabung pasangan simbol yang paling sering, berulang
- One-hot: `o_i` vektor panjang `V` (1 di posisi `i`); dot dua one-hot beda = 0
- Ambil embedding: `e_i = o_i E` ≡ ambil baris ke-`i` dari `E` (biaya `O(d)`)
- Alur tahap: `X = OE`, `O ∈ ℝ^{n×V}`, `E ∈ ℝ^{V×d}`, `X ∈ ℝ^{n×d}`
- Parameter embedding: `V × d`
- Weight tying: `logit = h Eᵀ` (berbagi matriks embedding, hemat `V×d` parameter)

### Bab 9 — Neuron, lapisan & fungsi aktivasi
- Neuron: `y = f(x · w + b)`; pre-aktivasi `z = x · w + b`
- Lapisan linear (dense): `y = f(xW + b)`, `W ∈ ℝ^{n×m}`, `b ∈ ℝ^m`
- Parameter lapisan: `n·m` bobot `+ m` bias
- Tanpa aktivasi: `(xW1)W2 = x(W1W2)` → dua lapisan = satu lapisan (kedalaman sia-sia)
- ReLU: `max(0, x)`; `ReLU′(x) = 1` jika `x > 0`, `0` jika `x < 0`
- Sigmoid: `σ(x) = 1/(1+e^{−x})`; `σ′ = σ(1−σ)` (maks 0,25)
- Tanh: `tanh′(x) = 1 − tanh²(x)`
- GELU: `x·Φ(x)`; hampiran `0,5x(1 + tanh[√(2/π)(x + 0,044715x³)])`
- SiLU/Swish: `x·σ(x)`
- SwiGLU: `SiLU(xW_gate) ⊙ (xW_up)` (perkalian elemen-per-elemen)
- FFN: `f(xW1 + b1)W2 + b2`, `W1 ∈ ℝ^{d×d_ff}`, `W2 ∈ ℝ^{d_ff×d}`, `d_ff = 4d`
- Parameter FFN: `2·d·d_ff + d_ff + d` (~2/3 parameter model tiap blok)
- FFN memproses tiap posisi terpisah; pertukaran antarposisi dilakukan attention

### Bab 10 — Softmax & fungsi kerugian
- Softmax: `softmax(z)_i = e^{z_i} / ∑_j e^{z_j}` (logit → probabilitas)
- Versi stabil: kurang dulu `max_j z_j` (invarian: `softmax(z+c) = softmax(z)`)
- Log-sum-exp: `LSE(z) = ln ∑_j e^{z_j} = max_j z_j + ln ∑_j e^{z_j−max}` (stabil)
- Loss satu token: `L = −ln p_c = LSE(z) − z_c`
- Gradien softmax + cross-entropy: `∂L/∂z = p − y` (selalu berjumlah 0)
- Sensitivitas skala: mengali logit dengan angka besar → distribusi runcing (one-hot)
- Gradien MSE (pembanding): `∂L_MSE/∂z_j = 2p_j[(p_j−y_j) − ∑_i(p_i−y_i)p_i]` → lenyap saat salah besar
- Label smoothing: `y_smooth = (1−ε)·y + ε/V`

### Bab 11 — Backpropagation
- Prinsip: aturan rantai dijalankan **mundur** (loss → masukan), hasil antara tetap vektor
- Error signal / delta: `δ^(ℓ) = ∂L/∂z^(ℓ)`
- Aturan mundur: `y=x+b` → `gout`; `y=xW` → `gin=gout·Wᵀ`, `grad_W=xᵀ·gout`
- ReLU: `gin = gout ⊙ 1[x>0]`; `y=a⊙b` → `gin=gout⊙b`
- Residual `y = x + F(x)`: `∇_x L = (I + J_F)ᵀ ∇_y L`; suku `I` lewat tanpa dikali
- Tanpa residual: gradien = hasil kali `ℓ` faktor (`0,8⁹⁶ ≈ 5·10⁻¹⁰` lenyap; `1,2⁹⁶ ≈ 4·10⁷` meledak)
- Batch: `Y = XW` → `grad_W = XᵀG`, `grad_X = G Wᵀ` (kontribusi semua token dijumlahkan)
- Biaya: backward ≈ 2× forward; satu langkah penuh ≈ 3× forward; `C ≈ 6·N·D` FLOP
- Memori aktivasi: `≈ k·L·n·d` (k ≈ 10–20) → gradient checkpointing

### Bab 12 — Optimizer
- Gradient descent: `θ ← θ − η·g`; batas stabil fungsi kuadrat `f=a(w−c)²`: `η < 1/a`
- Mini-batch: `g ≈ (1/B)∑∇L_i`; derau ∝ `1/√B`
- Momentum: `v ← βv + g`; `θ ← θ − η·v`; batas `v∞ = g/(1−β)` (β=0,9 → 10×)
- Adam: `m ← β1·m + (1−β1)g`; `v ← β2·v + (1−β2)g²` (kuadrat elemen-per-elemen)
- Koreksi bias: `m̂ = m/(1−β1^t)`, `v̂ = v/(1−β2^t)`; update `θ ← θ − η·m̂/(√v̂+ε)`
- Adam: langkah ≈ `η` (kebal penskalaan gradien per parameter); default `β1=0,9, β2=0,999, ε=1e-8`
- AdamW: `θ ← θ − η(m̂/(√v̂+ε) + λθ)` (weight decay terpisah, λ=0,1)
- Memori Adam: parameter+gradien+m+v = `16N` byte (7B → 112 GB)
- Schedule: warmup `η_t = η_max·t/T_w`, lalu cosine `η_t = η_min + ½(η_max−η_min)(1+cos(π(t−T_w)/(T−T_w)))`
- Gradient clipping: jika `‖g‖ > c` maka `g ← g·c/‖g‖` (c=1,0)

### Bab 13 — Attention (Query, Key, Value)
- Tiga peran dari vektor yang sama: `Q = HW_Q`, `K = HW_K`, `V = HW_V`
- Skor kecocokan: `S = QKᵀ` (`n×n`; baris = query, kolom = key)
- Bobot: `A = softmax(S/√d_k + M)` per baris
- Keluaran: `O = AV` (rata-rata berbobot dari value)
- Rumus lengkap: `Attention(Q,K,V) = softmax(QKᵀ/√d_k + M)·V`
- Skala `√d_k`: `Var(q·k) = d_k`, jadi dibagi `√d_k` → varians 1 (cegah softmax runtuh)
- Causal mask: `S''_ij = −∞` untuk `j > i` (pakai `−1e9` di kode); `e^{−∞}=0`
- Biaya: skor `2n²d_k`, softmax `≈3n²`, `AV 2n²d_v` (tumbuh `n²`); proyeksi `n²` → `n` di konteks panjang
- `∂L/∂W_Q = Hᵀ(∂L/∂Q)`; attention bisa dilatih

### Bab 14 — Multi-Head Attention
- `head_i = Attention(HW_Qᵢ, HW_Kᵢ, HW_Vᵢ)`; `MultiHead(H) = Concat(head_1,…,head_h)W_O`
- Tiap head di subruang `d_k = d/h` → jumlah parameter sama dengan 1 head besar
- Potong kolom matriks besar menjadi `h` blok; concat menyambung mendatar
- `W_O` (`h·d_v × d`): satu-satunya tempat hasil antarhead bercampur
- Header nyata: GPT-2 small `768/12→64`; Llama2-7B `4096/32→128`; GPT-3 `12288/96→128`
- KV cache: `2·L·h·n·d_k·byte`; MQA (1 K/V), GQA (`g` grup K/V) mengecilkan cache

### Bab 15 — Informasi posisi
- Attention **buta urutan** (equivarian permutasi) — tanpa causal mask, isi tak berubah saat baris diacak
- Learned PE: `H(0) = X + P`; parameter `n_max × d` (GPT-3: `2048×12288 = 25,2 jt`); panjang terbatas
- Sinusoidal: `PE(pos,2i) = sin(pos/10000^{2i/d})`, `PE(pos,2i+1) = cos(...)`; tanpa parameter
- RoPE: putar pasangan `(q_{2i}, q_{2i+1})` sebesar `mθ_i`, `θ_i = 10000^{−2i/d}`
  - `[q'_{2i}; q'_{2i+1}] = R(mθ_i)·[q_{2i}; q_{2i+1}]`; panjang tetap; diterapkan di tiap lapisan
  - `R(mθ)ᵀR(nθ) = R((n−m)θ)` → `q'_m·k'_n` hanya bergantung `m − n`
- ALiBi: `S_ij = q_i·k_j/√d_k − m_h·(i − j)` (penalti linear jarak, `m_h` tetap per head)
- Position interpolation: `pos/scale` (mis. 8192 → 2048) untuk memperpanjang konteks

### Bab 16 — Blok Transformer
- Satu blok (Pre-LN): `H' = H + MHA(LN(H))`; `H'' = H' + FFN(LN(H'))`
- Masuk `n×d`, keluar `n×d` → blok bisa ditumpuk tanpa ubah rumus
- Residual = jalur gradien tanpa dikali apa pun; normalisasi = kendali skala
- LayerNorm (per token): `γ·(h−μ)/√(σ²+ε) + β`; `μ,σ²` sepanjang dimensi fitur
- RMSNorm: `γ·h/√((1/d)∑h²+ε)` (tanpa mean, tanpa β; ε di dalam akar)
- Post-LN: `H' = LN(H + F(H))` (mini-LLM/Transformer asli); Pre-LN dipakai model modern
- Parameter per blok: `4d² + 2·d·d_ff + 4d` ≈ `12d²` (d_ff=4d); attention ⅓, FFN ⅔
- Total model: `N ≈ V·d + n_max·d + L·12d²`

### Bab 17 — Lapisan keluaran & perjalanan penuh
- Alur: token → embedding → +posisi → attention → FFN → logit → softmax → prediksi
- Lapisan keluaran (weight tying): `Z = H(L)·Eᵀ`, `Z_ij = h_i · e_j`
- Logit = dot product (dipengaruhi arah **dan** panjang: `h·e = ‖h‖‖e‖cos θ`)
- Logit `n×V` → prediksi **semua posisi** sekaligus; satu urutan `n` token → `n−1` contoh
- Jalur maju lengkap: `H(0)=E[x]+P`; `A=H+MHA(LN(H))`; `H=A+FFN(LN(A))`; `Z=LN(H)Eᵀ`; `p=softmax(z)`
- Ukuran tak berubah `n×d`; membesar hanya skor `n×n` dan FFN tengah `n×d_ff`
- Biaya maju: `≈ 2·N·n` FLOP (N parameter, n token)

### Bab 18 — Pretraining
- Objektif: rata-rata per token `L(θ) = −(1/∑|x|) ∑_x ∑_t ln P_θ(x_t | x_<t)`
- Label = teks itu sendiri (self-supervised); `input = tokens[:-1]`, `target = tokens[1:]`
- Satu urutan `n` token → `n − 1` sinyal loss (causal mask); token/langkah `= B × n`
- Gradient accumulation: `g = (1/k)∑ g_j` (memori aktivasi `/k`)
- Biaya: `C ≈ 6·N·D` FLOP; loss awal `≈ ln V`
- Scaling law: `L(N) ∝ N^{−0,076}`; `10× N` → loss `×0,84`
- Chinchilla: `D ≈ 20·N`; `L(N,D) = E + A/N^α + B/D^β` (E tak bisa direduksi)

### Bab 19 — Inference
- Greedy: `argmax p` (deterministik, cenderung berulang)
- Sampling murni: ekor panjang berisiko; temperature: `p_i ∝ e^{z_i/T}` (`T<1` runcing, `T>1` rata)
- Top-k: ambil `k` teratas lalu normalisasi (kandidat tetap)
- Top-p (nucleus): ambil sampai kumulatif `≥ p` (kandidat adaptif)
- Pipeline: `logit → /T → softmax → top-k → top-p → sample`
- KV cache: bagian attention per langkah `O(n²) → O(n)`; ukuran `2·L·n·d_kv·byte`
- Prefill (FLOP) vs decode (bandwidth memori): `≈ bandwidth/ukuran model` token/detik
- Beam search: lacak `b` jalur (bagus untuk terjemahan, buruk untuk chat)

### Bab 21 — Mini-LLM dari nol
- Konfigurasi: `V=8, d=4, L=1, head=1, d_ff=8`, panjang 7, Post-LN tanpa γ/β, ReLU, weight tying
- 188 parameter (184 dilatih; `P[6]` tak tersentuh); init `N(0, 0,5²)` seed 42
- Gradien **beda-hingga terpusat**, Adam `lr=0,05`, 400 langkah, float64
- Data: 4 kalimat × 6 = 24 pasangan; konteks `di atas` → 3× meja, 1× tikar
- Hasil: loss `2,4480 → 0,0955`; batas teoretis `0,0937`
- `P(meja)=0,7492`, `P(tikar)=0,2494` (data 0,75/0,25); cos(meja,tikar) `0,7557 → 0,9960`
- Attention mengerut jadi hampir one-hot; model belajar **distribusi**, bukan menghafal

### Bab 20 — SFT, RLHF, DPO, LoRA
- SFT: pretraining dilanjutkan pada instruksi–jawaban; loss **hanya token jawaban** `−(1/N_jawaban)∑_{t∈jawaban} ln P_θ(x_t|x_<t)`
- Reward model (Bradley–Terry): `P(y_w ≻ y_l) = σ(r_w − r_l)`; loss `−ln σ(r_w − r_l)`
- Reward: **pergeseran bebas, skala tidak** (σ hanya lihat selisih)
- RLHF: `max 𝔼[r_φ] − β·D_KL(π_θ‖π_ref)`; KL mencegah *reward hacking*; optimasi via PPO
- DPO: `L = −ln σ(β(Δln rasio_w − Δln rasio_l))`, `Δln rasio = ln π_θ(y) − ln π_ref(y)` (tanpa reward model/RL)
- LoRA: `W' = W + (α/r)·B·A`; `A` acak, `B` nol (mulai identik); latih `A,B` saja
- LoRA 4096×4096, r=8 → 65.536 param (0,39% dari full)
- Ketiga tahap: tulang punggung sama, beda data & bentuk loss

### Bab 22 — Peta besar
- Jalur maju lengkap: `H(0)=E[x]+P`; `H̃=LN(H)`; `Q,K,V=H̃W`; `A=softmax(QKᵀ/√d_k+M)`; `U=Concat(AV)W_O`; `G=H+U`; `H=G+act(LN(G)W1)W2`; `Z=LN(H(L))Eᵀ`; `p=softmax(Z)`
- Pelatihan: `L = −(1/(n−1))∑_{t=1}^{n−1} ln P(x_{t+1}|x_≤t)`; `θ ← θ − η·m̂/(√v̂+ε)`
- Sebaran parameter blok: attention `4d²` (⅓), FFN `8d²` (⅔)
- Parameter model: `N ≈ V·d + L·12d²` (d_ff=4d); contoh `d=1024,L=24,V=32000` → 334.757.888
