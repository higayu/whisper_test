# routes/recorder.py
from flask import Blueprint, render_template

bp = Blueprint("recorder", __name__)

@bp.get("/recorder")
def recorder():
    return render_template("recorder.html")
