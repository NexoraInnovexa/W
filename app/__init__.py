from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_mail import Mail
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import secrets
from flask_socketio import SocketIO
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
mail = Mail()
socketio = SocketIO()

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov'}

# Check if running on Render
USE_REDIS = os.getenv("RENDER", "false").lower() != "true"


if USE_REDIS:
    from redis import Redis
    redis_host = os.getenv('REDIS_HOST')
    redis_port = int(os.getenv('REDIS_PORT', 6379))  # Default to 6379
    redis_password = os.getenv('REDIS_PASSWORD')
    
    redis_client = Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True,
        ssl=True  # Required for Redis Cloud
    )
else:
    class RedisFallback:
        """Fallback class to simulate Redis behavior with a dictionary."""
        def __init__(self):
            self.store = {}

        def set(self, key, value):
            self.store[key] = value

        def get(self, key):
            return self.store.get(key)

        def delete(self, key):
            self.store.pop(key, None)
    
    redis_client = RedisFallback()
    print("⚠️ Running on Render: Using dictionary-based cache instead of Redis")


if not USE_REDIS:
    redis_client = RedisFallback()
    print("⚠️ Running on Render: Using dictionary-based cache instead of Redis")

# User loader for Flask-Login
from app.models import User
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Flask Configurations
    app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
    app.config['WTF_CSRF_ENABLED'] = True

    # Dynamically switch database based on environment
    # if os.getenv("RENDER") == "true":
    #     app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('POSTGRES_URI', 'postgresql://football_db_45ja_user:FcSz0jnwqUujnD1o1ZmWBaMEMP22RuiO@dpg-cv88815ds78s73e900hg-a/football_db_45ja')
    # else:
    #     app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    #         'LOCAL_POSTGRES_URI',  # Local Database
    #         'postgresql://postgres:Amarachi1994@localhost:5433/mydatabase'
    #     )

    # # Debug: Print the database URI being used
    # print(f"Using Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")


    # if not app.config.get("SQLALCHEMY_DATABASE_URI"):
    #     raise RuntimeError("Database URI is missing! Check your .env file.")

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'POSTGRES_URI', 
        'postgresql://football_db_45ja_user:FcSz0jnwqUujnD1o1ZmWBaMEMP22RuiO@dpg-cv88815ds78s73e900hg-a/football_db_45ja'
    )

    # Debugging: Log the database URI
    print(f"🔍 Using Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")

    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        raise RuntimeError("❌ Database URI is missing! Check your environment variables.")


    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config["PERMANENT_SESSION_LIFETIME"] = 3600  # 1-hour session expiration
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size

    # Flask-Mail configuration
    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
    app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
    app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    mail.init_app(app)
    socketio.init_app(app)

    # Register Blueprints (routes)
    from .routes import routes
    app.register_blueprint(routes)

    return app

# Create the app instance
app = create_app()

if __name__ == '__main__':
    socketio.run(app, debug=True)