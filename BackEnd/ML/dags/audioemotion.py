import speech_recognition as sr
import pyttsx3
import openai
from dotenv import load_dotenv
import os
from pathlib import Path
import torch
import torch.nn as nn
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")
LANGUAGE = "en-US"
EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
mic = sr.Microphone()

class EnhancedEmotionClassifier(nn.Module):
    def __init__(self, num_classes=len(EMOTION_LABELS)):
        super(EnhancedEmotionClassifier, self).__init__()
        input_features = 45  # 40 MFCC + 5 additional features
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        dummy_input = torch.randn(1, input_features, 216)
        with torch.no_grad():
            output_size = self.conv_layers(dummy_input).view(-1).size(0)
            
        self.fc_layers = nn.Sequential(
            nn.Linear(output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def extract_pitch_features(y, sr):
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0: 
                pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            return [0, 0, 0, 0, 0]  # Return zeros if no pitch detected
        
        pitch_values = np.array(pitch_values)
        
        # Calculate pitch statistics
        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)
        min_pitch = np.min(pitch_values)
        max_pitch = np.max(pitch_values)
        pitch_range = max_pitch - min_pitch
        
        return [mean_pitch, std_pitch, min_pitch, max_pitch, pitch_range]
    
    except Exception as e:
        print(f"Error extracting pitch features: {e}")
        return [0, 0, 0, 0, 0]

def extract_spectral_features(y, sr):
    """Extract spectral features that correlate with emotions"""
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        
        rms = librosa.feature.rms(y=y)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_rolloff': np.mean(spectral_rolloff),
            'zcr': np.mean(zcr),
            'rms': np.mean(rms),
            'tempo': tempo
        }
    except Exception as e:
        print(f"Error extracting spectral features: {e}")
        return {
            'spectral_centroid': 0,
            'spectral_rolloff': 0,
            'zcr': 0,
            'rms': 0,
            'tempo': 0
        }

def rule_based_emotion_analysis(pitch_stats, spectral_features):
    """Simple rule-based emotion detection using pitch and spectral features"""
    mean_pitch, std_pitch, min_pitch, max_pitch, pitch_range = pitch_stats
    
    if mean_pitch == 0:
        return "neutral"
    
    if mean_pitch > 200 and std_pitch > 20:  # High pitch with variation
        if spectral_features['rms'] > 0.02:  # High energy
            return "happy"
        else:
            return "fearful"
    elif mean_pitch > 180 and pitch_range > 100:  # High pitch with wide range
        return "surprised"
    elif mean_pitch < 120 and std_pitch < 15:  # Low pitch, stable
        if spectral_features['rms'] < 0.01:  # Low energy
            return "sad"
        else:
            return "calm"
    elif std_pitch > 25 and spectral_features['rms'] > 0.025:  # High variation, high energy
        return "angry"
    elif spectral_features['zcr'] < 0.05:  # Low zero crossing rate
        return "disgust"
    else:
        return "neutral"

def preprocess_audio_enhanced(audio_data):
    """Enhanced audio preprocessing with pitch and spectral features"""
    try:
        audio_array = np.frombuffer(audio_data.get_wav_data(), dtype=np.int16)
        y = librosa.util.normalize(audio_array.astype(float))
        sr = 16000
        
        if audio_data.sample_rate != sr:
            y = librosa.resample(y, orig_sr=audio_data.sample_rate, target_sr=sr)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Extract pitch features
        pitch_stats = extract_pitch_features(y, sr)
        
        # Extract spectral features
        spectral_features = extract_spectral_features(y, sr)
        
        # Resize MFCC to target size
        target_size = 5 * sr // 512 + 1  # ~216 frames for 5 seconds
        if mfccs.shape[1] > target_size:
            mfccs = mfccs[:, :target_size]
        else:
            padding = target_size - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, padding)), mode='constant')
        
        # Create additional feature arrays with same time dimension as MFCC
        pitch_features = np.array(pitch_stats).reshape(-1, 1)
        pitch_features = np.repeat(pitch_features, target_size, axis=1)
        
        # Combine all features
        combined_features = np.vstack([mfccs, pitch_features])
        
        return torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0), pitch_stats, spectral_features
        
    except Exception as e:
        print(f"Error during enhanced audio preprocessing: {e}")
        return None, [0, 0, 0, 0, 0], {}

def load_emotion_model(model_path='enhanced_emotion_model.pth'):
    """Load the enhanced emotion model"""
    model = EnhancedEmotionClassifier()
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print("Enhanced emotion model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Using untrained model.")
    else:
        print(f"Warning: Model file '{model_path}' not found. Using rule-based emotion detection.")
    return model

emotion_model = load_emotion_model()

def predict_emotion_enhanced(audio_tensor, pitch_stats, spectral_features):
    """Enhanced emotion prediction using both neural network and rules"""
    
    # Rule-based prediction as fallback
    rule_based_emotion = rule_based_emotion_analysis(pitch_stats, spectral_features)
    
    # Neural network prediction (if model is trained)
    if audio_tensor is not None:
        try:
            with torch.no_grad():
                outputs = emotion_model(audio_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Use neural network prediction if confidence is high
                if confidence.item() > 0.6:
                    return EMOTION_LABELS[predicted_idx.item()], confidence.item()
                else:
                    return rule_based_emotion, 0.5
        except Exception as e:
            print(f"Error in neural network prediction: {e}")
    
    return rule_based_emotion, 0.5

def analyze_pitch_emotion_correlation(pitch_stats, detected_emotion):
    """Analyze the correlation between pitch characteristics and detected emotion"""
    mean_pitch, std_pitch, min_pitch, max_pitch, pitch_range = pitch_stats
    
    analysis = f"\nPitch Analysis for '{detected_emotion}' emotion:\n"
    analysis += f"- Average Pitch: {mean_pitch:.1f} Hz\n"
    analysis += f"- Pitch Variation: {std_pitch:.1f} Hz\n"
    analysis += f"- Pitch Range: {min_pitch:.1f} - {max_pitch:.1f} Hz\n"
    
    # Emotional interpretation
    if mean_pitch > 200:
        analysis += "- High pitch suggests excitement, fear, or happiness\n"
    elif mean_pitch < 120:
        analysis += "- Low pitch suggests calmness, sadness, or authority\n"
    else:
        analysis += "- Moderate pitch suggests neutral or balanced emotional state\n"
        
    if std_pitch > 25:
        analysis += "- High pitch variation suggests emotional intensity\n"
    elif std_pitch < 10:
        analysis += "- Low pitch variation suggests emotional stability\n"
    
    return analysis

def ai_sentence_correction(raw_text):
    """Correct and format speech recognition output"""
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
    """Text-to-speech output"""
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen_and_respond():
    """Main function to listen to speech and respond with emotion analysis"""
    with mic as source:
        print("\nAdjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready! Say 'stop listening' to exit.\n")
        print("Features: Speech-to-Text + Pitch-based Emotion Analysis\n")

        while True:
            try:
                print("Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

                audio_tensor, pitch_stats, spectral_features = preprocess_audio_enhanced(audio)
                detected_emotion, confidence = predict_emotion_enhanced(audio_tensor, pitch_stats, spectral_features)
                
                print(f"Detected Emotion: {detected_emotion} (confidence: {confidence:.2f})")

                raw_text = recognizer.recognize_google(audio, language=LANGUAGE)
                print(f"Raw Speech: {raw_text}")

                if "stop listening" in raw_text.lower():
                    print("Exiting voice assistant...")
                    speak("Goodbye!")
                    break

                clean_text = ai_sentence_correction(raw_text)
                print(f"Corrected Text: {clean_text}")
                
                pitch_analysis = analyze_pitch_emotion_correlation(pitch_stats, detected_emotion)
                print(pitch_analysis)

                response_text = f"You sound {detected_emotion}. I heard: {clean_text}"
                print(f"Response: {response_text}")
                speak(response_text)

            except sr.WaitTimeoutError:
                print("No speech detected, still listening...")
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand you. Please speak more clearly.")
            except sr.RequestError as e:
                print(f"Speech recognition API error: {e}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
            
def process_uploaded_audio(file_path):
    """Process an uploaded audio file instead of live mic input."""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        pitch_stats = extract_pitch_features(y, sr)
        spectral_features = extract_spectral_features(y, sr)
        
        # Match the preprocessing format
        target_size = 5 * sr // 512 + 1  # ~216 frames
        if mfccs.shape[1] > target_size:
            mfccs = mfccs[:, :target_size]
        else:
            padding = target_size - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, padding)), mode='constant')
        
        pitch_features = np.array(pitch_stats).reshape(-1, 1)
        pitch_features = np.repeat(pitch_features, target_size, axis=1)
        
        combined_features = np.vstack([mfccs, pitch_features])
        audio_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)

        # Predict emotion
        detected_emotion, confidence = predict_emotion_enhanced(audio_tensor, pitch_stats, spectral_features)
        pitch_analysis = analyze_pitch_emotion_correlation(pitch_stats, detected_emotion)

        return {
            "emotion": detected_emotion,
            "confidence": confidence,
            "pitch_analysis": pitch_analysis
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Enhanced Voice Assistant with Pitch-based Emotion Analysis")
    print("=" * 60)
    listen_and_respond()