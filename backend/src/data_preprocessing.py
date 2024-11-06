# src/data_preprocessing.py
import json
import pandas as pd

def load_dataset(filepath='data/games.json'):
    """
    Ładuje dane z pliku JSON.

    :param filepath: Ścieżka do pliku JSON.
    :return: Słownik z danymi.
    """
    if not filepath:
        raise ValueError("Ścieżka do pliku jest pusta.")

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        return dataset
    except FileNotFoundError:
        raise FileNotFoundError(f"Plik {filepath} nie został znaleziony.")
    except json.JSONDecodeError:
        raise ValueError(f"Plik {filepath} zawiera niepoprawny format JSON.")

def parse_estimated_owners(owners_range):
    """
    Parsuje atrybut 'estimated_owners' z zakresu tekstowego na wartość liczbową.

    :param owners_range: Zakres właścicieli jako string, np. "0 - 20000".
    :return: Średnia wartość zakresu jako int lub None w przypadku błędu.
    """
    try:
        low, high = owners_range.split(' - ')
        return (int(low) + int(high)) // 2
    except (ValueError, AttributeError):
        return None  # W przypadku błędu zwróć None

def extract_features(dataset):
    """
    Ekstrahuje wybrane cechy z datasetu i zwraca DataFrame.

    :param dataset: Słownik z danymi aplikacji.
    :return: Pandas DataFrame z wybranymi cechami.
    """
    data = []
    for app, game in dataset.items():
        estimated_owners = parse_estimated_owners(game.get('estimated_owners', "0 - 0"))
        if estimated_owners is None:
            continue  # Pomijamy wpisy z błędnym formatem

        # Ekstrakcja cech numerycznych
        peak_ccu = game.get('peak_ccu', 0)
        price = game.get('price', 0.0)
        dlc_count = game.get('dlc_count', 0)
        positive = game.get('positive', 0)
        negative = game.get('negative', 0)
        average_playtime = game.get('average_playtime_forever', 0)
        required_age = game.get('required_age', 0)
        metacritic_score = game.get('metacritic_score', 0)
        user_score = game.get('user_score', 0)
        achievements = game.get('achievements', 0)
        recommendations = game.get('recommendations', 0)
        average_playtime_2weeks = game.get('average_playtime_2weeks', 0)

        # Ekstrakcja cech kategorycznych (przechowujemy je jako listy, nie jako ciągi znaków)
        languages = game.get('supported_languages', [])
        developers = game.get('developers', [])
        publishers = game.get('publishers', [])
        genres = game.get('genres', [])
        categories = game.get('categories', [])
        tags = game.get('tags', [])

        data.append([
            estimated_owners, peak_ccu, price, dlc_count, positive, negative,
            average_playtime, required_age, metacritic_score, user_score,
            achievements, recommendations, average_playtime_2weeks,
            languages, developers, publishers, genres, categories, tags
        ])

    columns = [
        'estimated_owners', 'peak_ccu', 'price', 'dlc_count', 'positive',
        'negative', 'average_playtime', 'required_age', 'metacritic_score',
        'user_score', 'achievements', 'recommendations', 'average_playtime_2weeks',
        'languages', 'developers', 'publishers', 'genres', 'categories', 'tags'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

def preprocess_data(df):
    """
    Przetwarza DataFrame: obsługuje brakujące wartości i przygotowuje dane do modelowania.

    :param df: Pandas DataFrame z wybranymi cechami.
    :return: Przetworzony DataFrame.
    """
    # Kolumny, które mogą zawierać listy
    columns_with_lists = ['languages', 'genres', 'developers', 'publishers', 'categories', 'tags']

    for col in columns_with_lists:
        # Jeśli kolumna zawiera listę, zamień ją na ciąg znaków
        df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Konwertowanie tags z dict na ciąg znaków
    df['tags'] = df['tags'].apply(lambda x: ', '.join(f"{k}: {v}" for k, v in x.items()) if isinstance(x, dict) else x)

    # Usuwanie duplikatów
    df = df.drop_duplicates()

    return df

def get_preprocessed_data(filepath='data/games.json'):
    """
    Kompleksowa funkcja do ładowania i przetwarzania danych.

    :param filepath: Ścieżka do pliku JSON.
    :return: Przetworzony Pandas DataFrame.
    """
    dataset = load_dataset(filepath)
    df = extract_features(dataset)
    df = preprocess_data(df)
    return df
