from app import create_app
from waitress import serve
import os

app = create_app()

for rule in app.url_map.iter_rules():
    print(rule.endpoint)

if __name__ == '__main__':
    env = os.getenv('FLASK_ENV', 'production')

    if env == 'development':
        print("Starting Flask App in development mode...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Starting Flask App in production mode with Waitress...")
        serve(app, host='0.0.0.0', port=5000)
