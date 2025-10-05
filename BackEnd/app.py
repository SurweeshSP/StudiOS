import os
import io
import sys
import time
import tempfile
import threading
import traceback
from dotenv import load_dotenv
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    Response,
    redirect,
)
from flask_cors import CORS

import cv2
from deepface import DeepFace
from transformers import pipeline

from config.db import init_db, db
from ML.dags.textemotion import predict_textemotion
from ML.dags.videoemotion import start_video_emotion
from ML.dags.audioemotion import process_uploaded_audio

import requests
import resend
import google.generativeai as genai

load_dotenv()
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_EMAIL = os.getenv("RESEND_EMAIL", "noreply@yourdomain.com")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure external libs
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# App setup
app = Flask(__name__, template_folder="templates")
CORS(app)

# If you have DB init function, initialize here
try:
    from config.db import init_db, db
    init_db(app)
    with app.app_context():
        db.session.execute("SELECT 1")
        print("Database connection OK")
except Exception as e:
    print("DB init or check skipped/failed:", e)


try:
    sentiment_analyzer = pipeline(
        task="text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        framework="pt",
    )
    print("Sentiment model loaded.")
except Exception as e:
    sentiment_analyzer = None
    print("Failed to load sentiment model:", e)

try:
    from ML.dags.textemotion import predict_textemotion  # optional
    print("Imported local predict_textemotion.")
except Exception:
    predict_textemotion = None

# Helper: audio processor (optional)
try:
    from ML.dags.audioemotion import process_uploaded_audio
    print("Imported local process_uploaded_audio.")
except Exception:
    process_uploaded_audio = None

class VideoCamera:

    def __init__(self, device=0, width=640, height=480):
        self.device = device
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.cap = None
        self.running = False

    def start(self):
        with self.lock:
            if self.running and self.cap is not None:
                return
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
            # fall back if not windows:
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.device)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.running = True
            print("Camera started.")

    def stop(self):
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self.running = False
            print("Camera stopped.")

    def read_frame(self):
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def generate_frames(self):
        """
        MJPEG frame generator for Flask Response.
        Analyze frames for emotion (DeepFace) and overlay text.
        """
        # ensure camera started
        if not self.running:
            self.start()

        try:
            while self.running:
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                try:
                    # DeepFace analysis (may be slow) - catch exceptions
                    result = DeepFace.analyze(img_path=frame, actions=["emotion"], enforce_detection=False)
                    dominant = result[0].get("dominant_emotion") if isinstance(result, list) else result.get("dominant_emotion")
                    if dominant:
                        cv2.putText(frame, f"Emotion: {dominant}", (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    # analysis failed - ignore and continue
                    # print("DeepFace error:", e)
                    pass

                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        finally:
            pass

camera = VideoCamera()


@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html")
    except Exception:
        return jsonify({"status": "ok", "message": "Flask server running"})


@app.route("/health", methods=["GET"])
def health():
    db_status = "unknown"
    try:
        if "db" in globals():
            with app.app_context():
                db.session.execute("SELECT 1")
            db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"
    return jsonify({"status": "healthy", "db": db_status})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        text = data.get("text")
        if not text:
            return jsonify({"error": "text required"}), 400

        # Prefer local predict_textemotion if available
        if predict_textemotion:
            result = predict_textemotion(text)
            return jsonify({"result": result})
        elif sentiment_analyzer:
            res = sentiment_analyzer(text)
            return jsonify({"result": res})
        else:
            return jsonify({"error": "No sentiment model available"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/analyze-text/<path:text>", methods=["GET"])
def analyze_text_route(text):
    try:
        if predict_textemotion:
            result = predict_textemotion(text)
            return jsonify(result)
        if sentiment_analyzer:
            result = sentiment_analyzer(text)
            return jsonify(result)
        return jsonify({"error": "No sentiment model available"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/video_feed")
def video_feed():
    try:
        camera.start()
        return Response(camera.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/start-video", methods=["GET"])
def start_video():
    try:
        camera.start()
        return jsonify({"status": "camera started", "stream": "/video_feed"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/stop-video", methods=["GET"])
def stop_video():
    try:
        camera.stop()
        return jsonify({"status": "camera stopped"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/audio", methods=["POST"])
def audio_upload():
    try:
        file = request.files.get("file")
        save_flag = request.form.get("save", "false").lower() == "true"
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Save temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".webm") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        emotion = None
        if process_uploaded_audio:
            try:
                emotion = process_uploaded_audio(tmp_path)
            except Exception as e:
                emotion = {"error": str(e)}

        if save_flag:
            final_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.stream.seek(0)
            with open(final_path, "wb") as f:
                f.write(open(tmp_path, "rb").read())
            return jsonify({"file": file.filename, "status": "saved", "emotion": emotion})

        return jsonify({"file": file.filename, "status": "cached", "emotion": emotion})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/send-email", methods=["POST"])
def send_email():
    if not RESEND_API_KEY:
        return jsonify({"error": "RESEND_API_KEY not configured"}), 500
    try:
        data = request.get_json(force=True)
        to_email = data.get("to")
        subject = data.get("subject", "Notification from STUDIOS")
        html = data.get("html", "<p>Hi — this is a notification.</p>")

        if not to_email:
            return jsonify({"error": "recipient required"}), 400

        r = resend.Emails.send({
            "from": RESEND_EMAIL,
            "to": to_email,
            "subject": subject,
            "html": html
        })
        return jsonify({"status": "sent", "response": str(r)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/github/login")
def github_login():
    if not GITHUB_CLIENT_ID:
        return jsonify({"error": "GITHUB_CLIENT_ID not configured"}), 500
    redirect_uri = request.args.get("redirect_uri") or ("http://localhost:3000/api/github/callback")
    auth_url = (
        "https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        "&scope=read:user user:email"
    )
    return redirect(auth_url)

@app.route("/api/github/callback")
def github_callback():
    if not (GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET):
        return jsonify({"error": "GitHub client not configured"}), 500
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "code missing"}), 400
    try:
        token_res = requests.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
            timeout=10
        )
        token_json = token_res.json()
        access_token = token_json.get("access_token")
        if not access_token:
            return jsonify({"error": "no access token", "details": token_json}), 400

        user_res = requests.get("https://api.github.com/user",
                                headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
        user_data = user_res.json()

        email_res = requests.get("https://api.github.com/user/emails",
                                 headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
        emails = email_res.json() if email_res.ok else []
        primary = next((e["email"] for e in emails if e.get("primary")), None)
        return jsonify({"profile": user_data, "primary_email": primary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot-gemini", methods=["POST"])
def chatbot_gemini():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not configured"}), 500
    try:
        data = request.get_json(force=True)
        user_text = data.get("text")
        if not user_text:
            return jsonify({"error": "text required"}), 400

        # optional local sentiment
        sentiment = None
        try:
            if predict_textemotion:
                sentiment = predict_textemotion(user_text)
            elif sentiment_analyzer:
                sentiment = sentiment_analyzer(user_text)
        except Exception:
            sentiment = None

        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"You are an empathetic assistant. User sentiment (local): {sentiment}. Message: {user_text}"
        response = model.generate_content(prompt)
        return jsonify({"user": user_text, "sentiment": sentiment, "bot": response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
