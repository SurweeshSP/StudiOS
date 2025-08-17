import os
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()

def init_db(app):
    DATABASE_URL = os.getenv("DATABASE_URL")

    if DATABASE_URL and "channel_binding=require" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("require&channel_binding=require", "require")

    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    return db
