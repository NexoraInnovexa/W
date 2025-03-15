# from flask import Flask
# from pymongo import MongoClient
# from .routes import routes


# def create_app():
#     app = Flask(__name__)
    
#     client = MongoClient('mongodb+srv://<Wink>:<wink>@cluster0.2vmbf.mongodb.net/')
#     db = client['W_app']  
#     app.config['DB'] = db

#     app.register_blueprint(routes)
#     return app

# app = create_app()

# if __name__ == '__main__':
#     app.run(debug=True)