from flask import Flask, render_template, request
import tensorflow as tf

# Kurangi penggunaan RAM/CPU TensorFlow di server
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

import numpy as np
import os
import glob
import uuid


app = Flask(__name__)


# Folder upload gambar
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# File yang diperbolehkan
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# Load model sekali saat server berjalan
model = load_model("cnn_apel.h5", compile=False)


# Kelas hasil prediksi
kelas = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]


# Nama tampil di website
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
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Tidak ada gambar yang diupload."
        )


    file = request.files["image"]


    if file.filename == "":
        return render_template(
            "index.html",
            error="Silakan pilih gambar terlebih dahulu."
        )


    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Format file tidak didukung. Gunakan PNG/JPG/JPEG."
        )


    # Hapus gambar lama agar storage tidak penuh
    files = glob.glob("static/uploads/*")

    for f in files:
        try:
            os.remove(f)
        except:
            pass



    # Buat nama file unik
    ext = file.filename.rsplit(".", 1)[1].lower()

    filename = f"{uuid.uuid4().hex}.{ext}"

    filename = secure_filename(filename)


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(filepath)



    try:

        # Sesuaikan dengan ukuran input model CNN
        img = image.load_img(
            filepath,
            target_size=(128,128)
        )


        img = image.img_to_array(img)

        img = img / 255.0


        img = np.expand_dims(
            img,
            axis=0
        )


        # Prediksi
        pred = model.predict(
            img,
            verbose=0
        )


        kelas_index = np.argmax(pred)


        hasil_asli = kelas[kelas_index]

        hasil = nama_tampil[hasil_asli]


        confidence = round(
            float(np.max(pred)) * 100,
            2
        )


        # Bersihkan session setelah prediksi
        tf.keras.backend.clear_session()



    except Exception as e:

        return render_template(
            "index.html",
            error=f"Gagal memproses gambar: {str(e)}"
        )



    return render_template(
        "results.html",
        hasil=hasil,
        confidence=confidence,
        gambar=filepath,
        deskripsi=keterangan[hasil_asli]
    )



if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )