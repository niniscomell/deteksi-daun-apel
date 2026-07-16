from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


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