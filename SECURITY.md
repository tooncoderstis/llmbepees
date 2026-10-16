# Kebijakan Keamanan

## Melaporkan kerentanan
Jangan buka issue publik untuk kerentanan. Kirim email ke pemelihara repo dengan langkah reproduksi. Kami menindaklanjuti secepatnya.

## Praktik
- Jangan commit rahasia (`.env` di-ignore).
- Rotasi kredensial yang pernah terekspos.
- Jalankan dependency audit & secret scan di CI (gitleaks).
- Ikuti OWASP ASVS untuk verifikasi keamanan layanan demo.
