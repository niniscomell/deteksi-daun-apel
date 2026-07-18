from flask import Flask, render_template, request,redirect
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

import numpy as np
import os
import uuid
from datetime import datetime


app = Flask(__name__)


# =========================
# DATABASE CONFIG
# =========================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)



# =========================
# DATABASE MODEL
# =========================

class PredictionHistory(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    gambar = db.Column(
        db.String(100)
    )

    hasil = db.Column(
        db.String(50)
    )

    confidence = db.Column(
        db.Float
    )

    tanggal = db.Column(
        db.String(50)
    )



# membuat tabel database
with app.app_context():
    db.create_all()



UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)



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

    "Apple___Apple_scab":
    "Apple Scab",

    "Apple___Black_rot":
    "Black Rot",

    "Apple___Cedar_apple_rust":
    "Cedar Apple Rust",

    "Apple___healthy":
    "Healthy"
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

    return render_template(
        "index.html"
    )

@app.route("/history")
def history():
    page = request.args.get(
        "page",
        1,
        type=int
    )
    
    search = request.args.get(
        "search",
        ""
    )


    if search:

        data = PredictionHistory.query.filter(
            PredictionHistory.hasil.contains(search)
        ).order_by(
            PredictionHistory.id.desc()
        ).paginate(
            page=page,
            per_page=5
        )

    else:

        data = PredictionHistory.query.order_by(
            PredictionHistory.id.desc()
        ).paginate(
            page=page,
            per_page=5
        )



    data_valid = []


    for item in data.items:

        filepath = os.path.join(
            "static/uploads",
            item.gambar
        )


        if os.path.exists(filepath):

            data_valid.append(item)



    return render_template(
        "history.html",
        data=data_valid,
        pagination=data,
        search=search
    )
@app.route("/delete/<int:id>")
def delete(id):

    data = PredictionHistory.query.get_or_404(id)


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        data.gambar
    )


    if os.path.exists(filepath):

        os.remove(filepath)



    db.session.delete(data)

    db.session.commit()



    return redirect("/history")

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
            float(np.max(pred))*100,
            2
        )




        # =========================
        # SIMPAN KE DATABASE
        # =========================


        data = PredictionHistory(

            gambar=filename,

            hasil=hasil,

            confidence=confidence,

            tanggal=datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        )


        db.session.add(data)

        db.session.commit()





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


@app.route("/cekdb")
def cekdb():

    data = PredictionHistory.query.all()

    hasil = ""

    for item in data:
        hasil += f"""
        ID: {item.id}<br>
        Gambar: {item.gambar}<br>
        Hasil: {item.hasil}<br>
        Confidence: {item.confidence}<br>
        Tanggal: {item.tanggal}<br>
        <hr>
        """

    return hasil

@app.route("/clean_history")
def clean_history():

    data = PredictionHistory.query.all()

    jumlah_hapus = 0

    for item in data:

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            item.gambar
        )

        if not os.path.exists(filepath):

            db.session.delete(item)
            jumlah_hapus += 1


    db.session.commit()


    return f"{jumlah_hapus} data history tanpa gambar berhasil dihapus"

if __name__ == "__main__":


    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )