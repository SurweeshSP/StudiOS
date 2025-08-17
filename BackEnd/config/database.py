from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

#USER SCHEMA
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    name = db.Column(db.String(100), nullable=True)

    text_inputs = db.relationship('TextInput', backref='user', lazy=True)
    audio_transcripts = db.relationship('AudioTranscript', backref='user', lazy=True)
    video_emotions = db.relationship('VideoEmotionLog', backref='user', lazy=True)
    social_media_inputs = db.relationship('SocialMediaInput', backref='user', lazy=True)
    emotion_logs = db.relationship('EmotionLog', backref='user', lazy=True)

#TEXT - CHAT TABLE
class TextInput(db.Model):
    __tablename__ = 'text_inputs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

#AUDIO FILE  TABLE
class AudioTranscript(db.Model):
    __tablename__ = 'audio_transcripts'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    audio_path = db.Column(db.String(255), nullable=False)
    transcript = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

#VIDEO TABLE
class VideoEmotionLog(db.Model):
    __tablename__ = 'video_emotion_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    video_path = db.Column(db.String(255), nullable=False)
    detected_emotion = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class SocialMediaInput(db.Model):
    __tablename__ = 'social_media_inputs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    platform = db.Column(db.String(50), nullable=False)   
    content = db.Column(db.Text, nullable=False)          
    post_url = db.Column(db.String(255), nullable=True)   
    author_handle = db.Column(db.String(100), nullable=True)  
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    subreddit = db.Column(db.String(100), nullable=True)
    title = db.Column(db.String(255), nullable=True)
    ups = db.Column(db.Integer, default=0)
    score = db.Column(db.Integer, default=0)
    num_comments = db.Column(db.Integer, default=0)
    created_utc = db.Column(db.DateTime, nullable=True)   
    subreddit_subscribers = db.Column(db.Integer, nullable=True)


class EmotionLog(db.Model):
    __tablename__ = 'emotion_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    source = db.Column(db.String(50), nullable=False)  
    emotion_label = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(100), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)