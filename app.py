from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import glob

app = Flask(__name__)

# Folder untuk menyimpan gambar upload
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model CNN
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # ✔ TAMBAHAN: hapus file lama biar storage tidak penuh
    files = glob.glob("static/uploads/*")
    for f in files:
        os.remove(f)

    # mengambil file gambar
    file = request.files["image"]

    # menyimpan gambar ke folder uploads
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

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

    return render_template(
        "results.html",
        hasil=hasil,
        confidence=confidence,
        gambar=filepath,
        deskripsi=keterangan[hasil_asli]
    )


if __name__ == "__main__":
    app.run(debug=True)