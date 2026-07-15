from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os
import glob
import uuid

app = Flask(__name__)

# Folder untuk menyimpan gambar upload
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ✔ TAMBAHAN: pastikan folder upload ada, kalau belum -> buat otomatis
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model CNN (cukup sekali saat server nyala, biar cepat)
model = load_model("cnn_apel.h5")

# Nama kelas sesuai hasil class_indices
kelas = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

# Nama yang ditampilkan di web
nama_tampil = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy"
}

# Deskripsi penyakit
keterangan = {
    "Apple___Apple_scab":
        "Apple Scab merupakan penyakit jamur yang menyebabkan bercak hitam pada daun dan buah apel.",

    "Apple___Black_rot":
        "Black Rot merupakan penyakit yang menyebabkan daun menghitam dan membusuk.",

    "Apple___Cedar_apple_rust":
        "Cedar Apple Rust merupakan penyakit jamur yang menyebabkan bercak kuning hingga jingga pada daun.",

    "Apple___healthy":
        "Daun apel dalam kondisi sehat dan tidak terdeteksi penyakit."
}


def allowed_file(filename):
    """✔ TAMBAHAN: cek apakah ekstensi file diizinkan (jaga-jaga upload file bukan gambar)"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # ✔ TAMBAHAN: validasi ada file yang dikirim
    if "image" not in request.files:
        return render_template("index.html", error="Tidak ada gambar yang diupload.")

    file = request.files["image"]

    # ✔ TAMBAHAN: validasi file tidak kosong
    if file.filename == "":
        return render_template("index.html", error="Silakan pilih gambar terlebih dahulu.")

    # ✔ TAMBAHAN: validasi tipe file (harus gambar)
    if not allowed_file(file.filename):
        return render_template("index.html", error="Format file tidak didukung. Gunakan PNG/JPG/JPEG.")

    # hapus file lama biar storage tidak penuh
    files = glob.glob("static/uploads/*")
    for f in files:
        os.remove(f)

    # ✔ PERBAIKAN: nama file dibuat aman & unik (hindari path traversal & bentrok nama)
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filename = secure_filename(filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # preprocessing gambar
        img = image.load_img(filepath, target_size=(128, 128))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # prediksi
        pred = model.predict(img)

        kelas_index = np.argmax(pred)

        hasil_asli = kelas[kelas_index]
        hasil = nama_tampil[hasil_asli]

        confidence = round(np.max(pred) * 100, 2)

    except Exception as e:
        # ✔ TAMBAHAN: tangkap error kalau gambar rusak/gagal diproses
        return render_template("index.html", error=f"Gagal memproses gambar: {str(e)}")

    return render_template(
        "results.html",
        hasil=hasil,
        confidence=confidence,
        gambar=filepath,
        deskripsi=keterangan[hasil_asli]
    )


if __name__ == "__main__":
    # ⚠ PENTING: debug=False saat deploy ke server publik
    # Port diambil dari environment variable (dikasih otomatis oleh platform hosting)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)