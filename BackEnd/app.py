from flask import Flask, request, jsonify, render_template
from ML.dags.textemotion import predict_textemotion
import os, sys
from config.db import init_db, db
from ML.dags.videoemotion import start_video_emotion
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), "ML","dags"))

app = Flask(__name__)

init_db(app)
print("DataBase connected successfullly...")


@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/chat', methods=["POST"])
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
        return jsonify({"status":"Video stream started.. Press q to exit"})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)