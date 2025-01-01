from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Function to preprocess data
def preprocess_data():
    # Load the data
    movies = pd.read_csv("C:/Users/Abdullah/Desktop/dataset/movies.csv")
    ratings = pd.read_csv("C:/Users/Abdullah/Desktop/dataset/ratings.csv")

    # Merge datasets
    merged_df = pd.merge(movies, ratings, on='movieId')

    # Calculate average rating for each movie
    avg_ratings = merged_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)

    # Merge average ratings with movies
    movies = pd.merge(movies, avg_ratings, on='movieId', how='left')

    # Handle missing values
    movies['avg_rating'] = movies['avg_rating'].fillna(movies['avg_rating'].mean())
    movies = movies.dropna(subset=['genres'])

    # One-hot encode genres
    genres = movies['genres'].str.get_dummies('|')

    # Combine features: genres and average rating
    features = pd.concat([genres, movies[['avg_rating']]], axis=1)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return movies, features_scaled

# Function to recommend movies
def recommend_movies_knn(target_movie, movies_df, features_scaled, k=5):
    # Find the index of the target movie
    target_movie_indices = movies_df[movies_df['title'].str.lower() == target_movie.lower()].index

    if len(target_movie_indices) == 0:
        return []

    target_index = target_movie_indices[0]

    # Fit k-NN
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    knn.fit(features_scaled)

    # Find neighbors
    distances, indices = knn.kneighbors([features_scaled[target_index]])

    # Extract recommended movie indices (excluding the target movie itself)
    recommended_indices = indices[0][1:]
    recommended_movies = movies_df.iloc[recommended_indices]['title'].tolist()

    return recommended_movies

# Preprocess data
movies, features_scaled = preprocess_data()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommendations', methods=['POST'])
def recommendations():
    movie_name = request.form.get('movie_name')  # Get movie name from form
    recommended_movies = recommend_movies_knn(movie_name, movies, features_scaled)

    if not recommended_movies:
        return render_template('recommendations.html', movie_name=movie_name, recommendations=["Movie not found!"])

    return render_template('recommendations.html', movie_name=movie_name, recommendations=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
