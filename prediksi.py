import joblib
import pandas as pd

# =====================================
# MEMBACA MODEL
# =====================================

model = joblib.load("model.pkl")

print("=" * 60)
print("      PREDIKSI TINGKAT KECANDUAN MEDIA SOSIAL")
print("=" * 60)

# =====================================
# INPUT DATA PENGGUNA
# =====================================

umur = int(input("Masukkan umur (tahun) : "))
durasi_sosmed = float(input("Durasi media sosial (jam/hari) : "))
screen_time = float(input("Screen time HP (jam/hari) : "))
frekuensi_buka = int(input("Frekuensi membuka media sosial per hari : "))
jumlah_platform = int(input("Jumlah platform media sosial yang digunakan : "))
jam_tidur = float(input("Jam tidur per malam : "))

print("\nAktivitas Fisik")
print("1 = Rendah")
print("2 = Sedang")
print("3 = Tinggi")

aktivitas_fisik = int(input("Pilih (1/2/3) : "))

# =====================================
# MEMBUAT DATA BARU
# =====================================

data_baru = pd.DataFrame({
    "umur": [umur],
    "durasi_sosmed": [durasi_sosmed],
    "screen_time": [screen_time],
    "frekuensi_buka": [frekuensi_buka],
    "jumlah_platform": [jumlah_platform],
    "jam_tidur": [jam_tidur],
    "aktivitas_fisik": [aktivitas_fisik]
})

# =====================================
# MELAKUKAN PREDIKSI
# =====================================

hasil = model.predict(data_baru)

# Jika model mendukung probabilitas
try:
    probabilitas = model.predict_proba(data_baru)
    nilai_prob = max(probabilitas[0]) * 100
except:
    nilai_prob = None

# =====================================
# MENAMPILKAN HASIL
# =====================================

print("\n" + "=" * 60)
print("                HASIL PREDIKSI")
print("=" * 60)

print("Tingkat Kecanduan :", hasil[0])

if nilai_prob is not None:
    print(f"Tingkat Keyakinan Model : {nilai_prob:.2f}%")

# =====================================
# SARAN
# =====================================

print("\nSaran :")

if hasil[0] == "Rendah":
    print("- Pertahankan penggunaan media sosial secara bijak.")
    print("- Tetap tidur 7–8 jam setiap malam.")
    print("- Lanjutkan aktivitas fisik secara rutin.")

elif hasil[0] == "Sedang":
    print("- Kurangi penggunaan media sosial sekitar 1–2 jam per hari.")
    print("- Hindari membuka media sosial sebelum tidur.")
    print("- Perbanyak aktivitas di luar ruangan.")
    print("- Aktifkan pengingat screen time.")

elif hasil[0] == "Tinggi":
    print("- Batasi screen time harian.")
    print("- Nonaktifkan notifikasi yang tidak penting.")
    print("- Hindari penggunaan media sosial sebelum tidur.")
    print("- Tingkatkan aktivitas fisik.")
    print("- Luangkan waktu untuk aktivitas tanpa gawai, seperti membaca atau berolahraga.")

print("=" * 60)
print("Terima kasih telah menggunakan sistem prediksi.")
print("=" * 60)