import pandas as pd
import requests


def get_game_list():
    url = 'https://steamspy.com/api.php?request=all'
    response = requests.get(url)
    data = response.json()
    return data


def get_game_details(appid):
    url = f'https://steamspy.com/api.php?request=appdetails&appid={appid}'
    response = requests.get(url)
    data = response.json()
    return data


# Pobieramy listę wszystkich gier
all_games = get_game_list()

# Tworzymy pustą listę na szczegóły gier
games_details = []

# Limitujemy liczbę gier dla przykładu (np. 1000 gier)
for i, appid in enumerate(all_games.keys()):
    if i >= 50000:
        break
    game_data = get_game_details(appid)
    games_details.append(game_data)

print("Games are parsed")

# Zapisz dane do pliku JSON
import json

with open('game_details.json', 'w') as f:
    json.dump(games_details, f)
exit(1)

# Konwertujemy listę do DataFrame
df = pd.DataFrame(games_details)

# Wybieramy interesujące kolumny
df = df[['name', 'developer', 'publisher', 'positive', 'negative', 'owners', 'price', 'tags', 'genre']]

# Usuwamy wiersze z brakującymi wartościami
df.dropna(inplace=True)

# Konwertujemy kolumny numeryczne do odpowiednich typów
numeric_cols = ['positive', 'negative', 'price']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Usuwamy wiersze z błędnymi wartościami po konwersji
df.dropna(subset=numeric_cols, inplace=True)


# Przetwarzamy kolumnę 'owners' na wartość liczbową (średnia z przedziału)
def parse_owners(owners_str):
    min_owner, max_owner = owners_str.replace(',', '').split(' .. ')
    return (int(min_owner) + int(max_owner)) / 2


df['owners'] = df['owners'].apply(parse_owners)

# Przetwarzamy kolumny kategoryczne
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Kodowanie 'developer' i 'publisher'
le_dev = LabelEncoder()
df['developer_encoded'] = le_dev.fit_transform(df['developer'])

le_pub = LabelEncoder()
df['publisher_encoded'] = le_pub.fit_transform(df['publisher'])

# Przetwarzanie 'tags' i 'genre' na listy
df['tags_list'] = df['tags'].apply(lambda x: list(x.keys()) if x else [])
df['genre_list'] = df['genre'].apply(lambda x: x.split(',') if x else [])

# One-hot encoding dla 'tags' i 'genre'
mlb_tags = MultiLabelBinarizer()
tags_encoded = mlb_tags.fit_transform(df['tags_list'])
tags_df = pd.DataFrame(tags_encoded, columns=mlb_tags.classes_)

mlb_genre = MultiLabelBinarizer()
genre_encoded = mlb_genre.fit_transform(df['genre_list'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb_genre.classes_)

# Łączymy zakodowane dane z oryginalnym DataFrame
df = pd.concat([df, tags_df, genre_df], axis=1)

# Wybieramy cechy do modelu
feature_columns = [
                      'positive', 'negative', 'owners',
                      'developer_encoded', 'publisher_encoded'
                  ] + list(tags_df.columns) + list(genre_df.columns)

X = df[feature_columns]
y = df['price']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja i trenowanie modelu
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)

# Ocena modelu
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

import joblib

# Zapisywanie modelu
joblib.dump(model, '../ml_model/game_price_model.pkl')

# Zapisywanie enkoderów
joblib.dump(le_dev, '../ml_model/encoders/developer_encoder.pkl')
joblib.dump(le_pub, '../ml_model/encoders/publisher_encoder.pkl')
joblib.dump(mlb_tags, '../ml_model/encoders/tags_encoder.pkl')
joblib.dump(mlb_genre, '../ml_model/encoders/genre_encoder.pkl')
