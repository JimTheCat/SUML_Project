from flask import Flask

from routes import api_blueprint

app = Flask(__name__)

# Rejestracja blueprintów
app.register_blueprint(api_blueprint, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
