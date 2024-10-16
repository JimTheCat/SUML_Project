# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Ładowanie wytrenowanego modelu
MODEL_PATH = 'model/random_forest_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model załadowany pomyślnie.")
else:
    model = None
    print(f"Model nie został znaleziony pod ścieżką: {MODEL_PATH}")

# Funkcja do parsowania 'estimated_owners' (jeśli potrzebna)
def parse_estimated_owners(owners_range):
    try:
        low, high = owners_range.split(' - ')
        return (int(low) + int(high)) // 2
    except:
        return None

# Funkcja do przygotowania danych wejściowych
def prepare_data(input_data):
    """
    Konwertuje dane wejściowe na format używany przez model.

    :param input_data: Słownik z danymi nowej gry.
    :return: Przetworzony DataFrame.
    """
    # Ekstrakcja cech
    data = {
        'peak_ccu': input_data.get('peak_ccu', 0),
        'price': input_data.get('price', 0.0),
        'dlc_count': input_data.get('dlc_count', 0),
        'positive': input_data.get('positive', 0),
        'negative': input_data.get('negative', 0),
        'average_playtime': input_data.get('average_playtime', 0),
        'required_age': input_data.get('required_age', 0),
        'metacritic_score': input_data.get('metacritic_score', 0),
        'user_score': input_data.get('user_score', 0),
        'achievements': input_data.get('achievements', 0),
        'recommendations': input_data.get('recommendations', 0),
        'average_playtime_2weeks': input_data.get('average_playtime_2weeks', 0),
        'languages': input_data.get('languages', ""),
        'developers': input_data.get('developers', ""),
        'publishers': input_data.get('publishers', ""),
        'genres': input_data.get('genres', ""),
        'categories': input_data.get('categories', ""),
        'tags': input_data.get('tags', "")
    }

    # Konwersja do DataFrame
    df = pd.DataFrame([data])
    return df

# Endpoint do przewidywania liczby właścicieli gry
@app.route('/predict', methods=['POST'])
def predict():
    # Oczekujemy danych w formacie JSON
    if request.method == 'POST':
        try:
            input_data = request.json
            if not input_data:
                return jsonify({'error': 'Brak danych wejściowych'}), 400

            data = prepare_data(input_data)

            # Sprawdzenie, czy model jest załadowany
            if model is None:
                return jsonify({'error': 'Model is not loaded!'}), 500

            # Dokonaj przewidywania
            prediction = model.predict(data)[0]

            # Zwrócenie przewidywania w formacie JSON
            return jsonify({
                'predicted_owners': int(prediction)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

# Endpoint testowy
@app.route('/', methods=['GET'])
def index():
    return "Game Ownership Prediction API is running!"

if __name__ == '__main__':
    app.run(debug=True)
