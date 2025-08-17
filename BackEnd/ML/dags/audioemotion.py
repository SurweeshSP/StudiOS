import speech_recognition as sr
import pyttsx3
import openai
from dotenv import load_dotenv
import os
from pathlib import Path
import deeplake
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")
LANGUAGE = "en-US"
EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
mic = sr.Microphone()

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=len(EMOTION_LABELS)):
        super(EmotionClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        dummy_input = torch.randn(1, 40, 216) # Assuming an input size of 216
        with torch.no_grad():
            output_size = self.conv_layers(dummy_input).view(-1).size(0)
            
        self.fc_layers = nn.Sequential(
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def load_emotion_model(model_path='emotion_model.pth'):
    model = EmotionClassifier()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Emotion model loaded successfully.")
    else:
        print(f"Warning: Model file '{model_path}' not found. Using an untrained model.")
    return model

emotion_model = load_emotion_model()

def preprocess_audio(audio_data):
    try:
        audio_array = np.frombuffer(audio_data.get_wav_data(), dtype=np.int16)
        y = librosa.util.normalize(audio_array.astype(float))
        sr = 16000
        if audio_data.sample_rate != sr:
            y = librosa.resample(y, orig_sr=audio_data.sample_rate, target_sr=sr)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        target_size = 5 * sr // 512 + 1 
        if mfccs.shape[1] > target_size:
            mfccs = mfccs[:, :target_size]
        else:
            padding = target_size - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, padding)), mode='constant')
            
        return torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0) # Add batch dimension
    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        return None

def predict_emotion(audio_tensor):
    """Predicts emotion from preprocessed audio tensor."""
    if audio_tensor is None:
        return "Unknown"
    
    with torch.no_grad():
        outputs = emotion_model(audio_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        return EMOTION_LABELS[predicted_idx.item()]

def ai_sentence_correction(raw_text):
    try:
        prompt = f"Rewrite the following spoken words into a clean, properly formatted English sentence:\n\n'{raw_text}'\n\nCorrected sentence:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        return raw_text

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen_and_respond():
    with mic as source:
        print("\nAdjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready! Say 'stop listening' to exit.\n")

        while True:
            try:
                print("Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

                audio_tensor = preprocess_audio(audio)
                detected_emotion = predict_emotion(audio_tensor)
                print(f"Detected Emotion: {detected_emotion}")

                raw_text = recognizer.recognize_google(audio, language=LANGUAGE)
                print(f"Raw Speech: {raw_text}")

                if "stop listening" in raw_text.lower():
                    print("Exiting voice assistant...")
                    speak("Goodbye!")
                    break

                clean_text = ai_sentence_correction(raw_text)
                print(f"Corrected: {clean_text}")

                response_text = f"You sound {detected_emotion}. {clean_text}"
                speak(response_text)

            except sr.WaitTimeoutError:
                print("No speech detected, still listening...")
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand you.")
            except sr.RequestError as e:
                print(f"API Error: {e}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    listen_and_respond()