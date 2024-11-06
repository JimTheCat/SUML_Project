# src/model_training.py

import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

from data_preprocessing import get_preprocessed_data

def train_model():
    # Ładowanie i przygotowanie danych
    df = get_preprocessed_data('../data/games.json')

    print("Dane po przetworzeniu:")
    print(df.head())

    # Przygotowanie cech i etykiety
    X = df.drop('estimated_owners', axis=1)
    y = df['estimated_owners']

    # Podział na zestaw treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Definicja cech kategorycznych i numerycznych
    categorical_features = ['languages', 'developers', 'publishers', 'genres', 'categories', 'tags']
    numerical_features = [
        'peak_ccu', 'price', 'dlc_count', 'positive', 'negative',
        'average_playtime', 'required_age', 'metacritic_score',
        'user_score', 'achievements', 'recommendations', 'average_playtime_2weeks'
    ]

    # Pipeline dla cech numerycznych
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Sprawdzamy unikalne kategorie w kolumnach kategorycznych w zbiorze treningowym i testowym
    print("Unikalne kategorie w zbiorze treningowym:")
    for col in categorical_features:
        print(f"{col}: {X_train[col].unique()}")

    print("Unikalne kategorie w zbiorze testowym:")
    for col in categorical_features:
        print(f"{col}: {X_test[col].unique()}")


    # Pipeline dla cech kategorycznych
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Łączymy wszystkie w ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    # Pipeline modelu
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=15, random_state=42))
    ])

    # Trenowanie modelu
    print("Trenowanie modelu...")
    model_pipeline.fit(X_train, y_train)
    print("Model wytrenowany pomyślnie.")

    # Prognozowanie na zestawie testowym
    print("Prognozowanie na zestawie testowym...")
    y_pred = model_pipeline.predict(X_test)
    print("Prognozowanie zakończone.")

    # Ocena modelu
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

    # Zapisanie modelu
    joblib.dump(model_pipeline, '../model/random_forest_model.pkl')
    print("Model zapisany jako 'model/random_forest_model.pkl'")

if __name__ == '__main__':
    train_model()
