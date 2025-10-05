from fastapi import requests
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import threading, cv2
from deepface import DeepFace
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load model once
sentiment_analyzer = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    framework="pt",
)

def analyze_text_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = cap.read()
        if not success:
            break
        try:
            result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Video streaming route
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Restore /start-video route for backward compatibility
@app.route("/start-video", methods=["GET"])
def start_video_alias():
    return jsonify({"status": "Use /video_feed for streaming"})

@app.route("/analyze-text/<text>", methods=["GET"])
def analyze_text(text):
    sentiment = analyze_text_sentiment(text)

    return jsonify(sentiment)

CLIENT_ID = "Ov23ligkzwQ4k2mhI40y"
CLIENT_SECRET = "73915d3b78fc3b0af4cdcfa0e3a7472edcd64876"

@app.route("/api/github/callback")
def github_callback():
    code = request.args.get("code")
    token_res = requests.post(
        "https://github.com/login/oauth/access_token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
        },
        headers={"Accept": "application/json"}
    )
    token_json = token_res.json()
    access_token = token_json.get("access_token")

    # Fetch GitHub user info
    user_res = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    user_data = user_res.json()

    # Example response
    return jsonify(user_data)

if __name__ == "__main__":
    app.run(port=5000)
