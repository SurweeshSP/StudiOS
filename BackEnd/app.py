import os,io, tempfile
import sys
import threading
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from sqlalchemy import text
from config.db import init_db, db
from ML.dags.textemotion import predict_textemotion
from ML.dags.videoemotion import start_video_emotion
from ML.dags.audioemotion import process_uploaded_audio


# Load env variables
load_dotenv()

# Ensure ML path is available
sys.path.append(os.path.join(os.path.dirname(__file__), "ML", "dags"))

app = Flask(__name__)

# Initialize DB
init_db(app)

# Check DB connection at startup
with app.app_context():
    try:
        db.session.execute(text("SELECT 1"))
        print("Database connection successful")
    except Exception as e:
        print("Database connection failed:", e)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def text():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Pls provide a text"}), 400

        text = data["text"]
        result = predict_textemotion(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video", methods=["GET"])
def video():
    try:
        thread = threading.Thread(target=start_video_emotion)
        thread.start()
        return jsonify({"status": "Video stream started.. Press q to exit"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Health check route
@app.route("/health", methods=["GET"])
def health_check():
    try:
        db.session.execute(text("SELECT 1"))
        return jsonify({"status": "healthy", "db": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "db_error": str(e)}), 500

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
audio_cache = None
audio_emotion = None

@app.route("/audio", methods=["POST"])
def handle_audio():
    global audio_cache, audio_emotion

    file = request.files.get("file")
    save_flag = request.form.get("save", "false").lower() == "true"

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Read into memory
    audio_cache = file.read()

    # Save to a temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_cache)
        tmp_path = tmp.name

    try:
        from ML.dags.audioemotion import process_uploaded_audio
        audio_emotion = process_uploaded_audio(tmp_path)   # <-- pass path instead of bytes
    except Exception as e:
        audio_emotion = {"error": str(e)}

    # Save permanently if requested
    if save_flag:
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(audio_cache)
        return jsonify({
            "file": file.filename,
            "status": "saved",
            "emotion": audio_emotion
        })

    return jsonify({
        "file": file.filename,
        "status": "cached",
        "emotion": audio_emotion
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
