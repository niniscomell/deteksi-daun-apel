from flask import Flask, render_template, request
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

import numpy as np
import os
import glob
import uuid


app = Flask(__name__)


UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# LOAD TFLITE MODEL
# =========================

interpreter = Interpreter(
    model_path="cnn_apel.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# =========================
# KELAS
# =========================

kelas = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]


nama_tampil = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy"
}


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
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS



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
            error="Silakan pilih gambar."
        )


    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Format file tidak didukung."
        )


    # hapus gambar lama
    files = glob.glob("static/uploads/*")
    for f in files:
        os.remove(f)



    ext = file.filename.rsplit(".",1)[1].lower()

    filename = secure_filename(
        f"{uuid.uuid4().hex}.{ext}"
    )


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(filepath)



    try:

        img = image.load_img(
            filepath,
            target_size=(128,128)
        )


        img = image.img_to_array(img)

        img = img / 255.0


        img = np.expand_dims(
            img,
            axis=0
        ).astype(np.float32)



        # =====================
        # TFLITE PREDICTION
        # =====================

        interpreter.set_tensor(
            input_details[0]["index"],
            img
        )


        interpreter.invoke()


        pred = interpreter.get_tensor(
            output_details[0]["index"]
        )


        kelas_index = np.argmax(pred)


        hasil_asli = kelas[kelas_index]

        hasil = nama_tampil[hasil_asli]


        confidence = round(
            float(np.max(pred)) * 100,
            2
        )


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
        os.environ.get("PORT",5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )