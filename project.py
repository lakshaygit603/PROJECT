import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("movies.csv", low_memory=False)

df = df[['title', 'original_language', 'release_date', 'director', 'certification']]

df.dropna(subset=['title', 'original_language', 'release_date'], inplace=True)
df['director'].fillna('Unknown', inplace=True)
df['certification'].fillna(df['certification'].mode()[0], inplace=True)  

df['title'] = df['title'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)

df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['release_year'].fillna(df['release_year'].median(), inplace=True)

df['Movie Code'] = df['title'].factorize()[0]
df['Director Code'] = df['director'].factorize()[0]
df['Certification Code'] = df['certification'].factorize()[0] 

X = df[['Movie Code', 'release_year', 'Director Code', 'Certification Code']]
y = df['original_language']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

def predict_language(movie_name):
    movie_name = movie_name.lower().replace(r'[^a-z0-9 ]', '', regex=True)
    movie_info = df[df['title'] == movie_name]
    
    if not movie_info.empty:
        movie_code = movie_info['Movie Code'].values[0]
        release_year = movie_info['release_year'].values[0]
        director_code = movie_info['Director Code'].values[0]
        cert_code = movie_info['Certification Code'].values[0]
        director_name = movie_info['director'].values[0]
        certification = movie_info['certification'].values[0]
        
        movie_data = np.array([[movie_code, release_year, director_code, cert_code]])
        predicted_language = model.predict(movie_data)[0]
        
        return predicted_language, certification, release_year, director_name
    else:
        return "Unknown movie", "N/A", "N/A", "N/A"

movie_input = input("Enter a movie name: ")
predicted_language, movie_cert, movie_year, movie_director = predict_language(movie_input)

print(f"The predicted language for '{movie_input}' is: {predicted_language}\nCertification: {movie_cert}\nYear of Release: {movie_year}\nDirector: {movie_director}")
