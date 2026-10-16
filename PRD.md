# PRD: llmbepees

**Status**: dikembangkan — Fase 0 (Setup & Fondasi) dikerjakan.
**Versi dokumen**: 0.1
**Terkait**: `STATUS.md`, `CHANGELOG.md`, `docs/adr/`

> Dokumen ini menjawab **apa & kenapa**. Progres ada di `STATUS.md`, riwayat rilis di `CHANGELOG.md`, keputusan teknis di `docs/adr/`.

## 1. Ringkasan Produk
`llmbepees` adalah proyek **asisten domain yang faktual & anti-halusinasi**, dibangun dari nol. Alih-alih mengandalkan hafalan bobot model kecil, jawaban **di-grounding** ke basis dokumen: sistem mengambil potongan dokumen relevan, lalu model menjawab **hanya dari konteks** itu — dan **abstain** ("tidak tahu") bila tidak ada. Semuanya memakai komponen yang kita bangun sendiri (tokenizer BPE, model GPT PyTorch, embedding, loss, serving).

Starter proyek (struktur paket, uji, CI, deploy, hub, dan catatan belajar 22 bab) **disalin dari `aasallm`** lalu diganti nama — lihat [ADR-0008](docs/adr/0008-starter-dari-aasallm.md).

Target pengguna: pemilik dokumen yang ingin **tanya-jawab akurat** atas datanya sendiri, serta pembelajar yang ingin melihat mekanismenya transparan.

## 2. Prinsip Desain Utama
- **Grounding dulu.** Faktualitas berasal dari pencarian & kutipan dokumen, bukan dari bobot model.
- **Abstain lebih baik daripada mengarang.** Jika konteks tidak memuat jawaban → katakan tidak tahu.
- **Dari nol & transparan.** Tokenizer, model, embedding, loss, dan retrieval kita sendiri (NumPy sebagai rujukan, PyTorch untuk latih).
- **CPU-first.** Harus jalan & dilayani di CPU (EasyPanel), tanpa GPU.
- **Terverifikasi.** Akurasi diukur dengan set evaluasi berlabel (faktual, abstain, halusinasi).

## 3. Peran Pengguna (Roles)
| Role | Deskripsi | Akses |
|---|---|---|
| Pemilik dokumen | Menyediakan dokumen domain & set evaluasi | Basis pengetahuan |
| Pengguna demo | Tanya-jawab via halaman chat/API | Publik (rate-limited) |
| Operator | Melatih, mendeploy, memantau | MCP/CLI EasyPanel, `.env` lokal |

## 4. Fase / Modul Pengembangan
```
FASE 0  → Setup & Fondasi (starter dari aasallm) — dikerjakan
FASE 1  → Tokenizer BPE & dataset domain
FASE 2  → Model sedang + pretraining domain
FASE 3  → SFT (instruksi/tanya-jawab)
FASE 4  → Grounding (retrieval/RAG) + abstain
FASE 5  → Evaluasi faktual & anti-halusinasi
FASE 6  → Integrasi layanan + deploy CPU
BACKLOG → DPO/penyelarasan, kuantisasi lanjutan, multimodal, dll.
```

### 4.4 Grounding (retrieval)
- **Deskripsi**: indeks dokumen (chunk + embedding sendiri; fallback BM25), ambil top-k, susun prompt "jawab hanya dari konteks".
- **Acceptance**: /explain menampilkan konteks & skor retrieval; abstain saat skor < ambang.

### 4.5 Evaluasi
- **Deskripsi**: set eval (pertanyaan berlabel: jawaban benar / harus-abstain).
- **Acceptance**: akurasi faktual ≥80%, halusinasi ≤5%, abstain benar ≥90% (target awal; disesuaikan).

### 4.6 Integrasi & deploy
- **Deskripsi**: backend `GroundedEngine` di `llmbepees.serve`, endpoint tetap (`/health`, `/generate`, `/explain`, halaman chat).
- **Acceptance**: `GET /health = 200` di domain EasyPanel; jawaban mengutip konteks.

## 5. Model Data (ERD)
Bukan aplikasi ber-basis data. Artefak: **dokumen sumber**, **indeks KB** (`.npz`), **tokenizer**, **checkpoint model**, **set evaluasi**, **laporan eval**.

## 6. Non-Functional Requirements
- **Akurasi terukur**: metrik eksplisit pada set eval.
- **Latensi wajar di CPU** (rate-limited).
- **Keamanan**: `.env` di-ignore; secret scan (gitleaks) di CI; dokumen privat tidak dikomit.
- **Portabilitas**: NumPy baseline; PyTorch opsional; tokenizer & retrieval sendiri.

## 7. KPI & Metrik Keberhasilan
| Area | Metrik | Target |
|---|---|---|
| Faktualitas | Akurasi pada set eval | ≥ 80% |
| Anti-halusinasi | Jawaban keliru tanpa dasar konteks | ≤ 5% |
| Abstain | Benar abstain saat tak ada konteks | ≥ 90% |
| Retrieval | Recall@k relevan | ≥ 85% |
| Operasional | `/health` layanan | 200 |

## 8. Kepatuhan & Keamanan
- Repo publik; **tanpa** dokumen privat, data pribadi, atau kredensial.
- Dokumen domain privat dibangun jadi indeks **lokal** / via volume; tidak di-commit.
- `.env` tidak di-commit; secret scan di CI.

## 9. Risiko & Mitigasi
| # | Risiko | Dampak | Mitigasi |
|---|---|---|---|
| 1 | Model kecil halusinasi | Tinggi | Grounding wajib + abstain + temperature rendah |
| 2 | Dokumen privat bocor ke repo | Tinggi | `.gitignore`, indeks tak dikomit, hanya contoh publik |
| 3 | Latensi CPU | Sedang | model sedang, konteks pendek, kuantisasi bila perlu |
| 4 | Data SFT sedikit | Sedang | sintesis Q/A dari dokumen + augmentasi |
| 5 | Retrieval meleset (recall rendah) | Sedang | hybrid embedding + BM25, chunking & metadata |

## 10. Out of Scope
- Asisten umum luas (di luar jangkauan model kecil di CPU).
- Pelatihan skala produksi (butuh GPU + data besar).
- Multimodal, MoE, infrastruktur terdistribusi (BACKLOG).

## 11. Ringkasan Keputusan
| # | Pertanyaan | Keputusan |
|---|---|---|
| 1 | Tujuan | Asisten domain faktual & anti-halusinasi (Indonesia) |
| 2 | Pendekatan | Model sendiri + SFT + grounding (RAG) |
| 3 | Kompute | CPU-only (EasyPanel) |
| 4 | Starter | Mirror penuh dari `aasallm`, paket di-rename `llmbepees` (ADR-0008) |
| 5 | Deploy | EasyPanel project `m`, service `llmbepeesweb` (ADR-0007) |
