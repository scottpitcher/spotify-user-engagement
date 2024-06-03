import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from surprise import Dataset, Reader, SVD
from sklearn.model_selection import train_test_split
import joblib

# Function to authenticate and retrieve playlist tracks
def get_spotify_tracks(playlist_id, sp):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

# Load Spotify API credentials
def authenticate_spotify():
    client_id = 'YOUR_SPOTIPY_CLIENT_ID'
    client_secret = 'YOUR_SPOTIPY_CLIENT_SECRET'
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
    return sp

# Generate synthetic data for user interactions
def generate_synthetic_data(sp):
    playlists = {
        "Today's Top Hits": "37i9dQZF1DXcBWIGoYBM5M",
        "Global Top 50": "37i9dQZEVXbMDoHDwVN2tF",
        "Global Viral 50": "37i9dQZEVXbLiRSasKsNU9",
        "New Music Friday": "37i9dQZF1DX4JAvHpjipBk",
        # Add other playlists as needed
    }
    
    song_data = []
    for playlist_name, playlist_id in playlists.items():
        tracks = get_spotify_tracks(playlist_id, sp)
        for song in tracks:
            track = song['track']
            song_data.append({
                'song_id': track['id'],
                'title': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'duration': track['duration_ms'] // 1000,  # Convert ms to seconds
                'popularity': track['popularity'],
                'release_date': track['album']['release_date']
            })
    
    song_df = pd.DataFrame(song_data)
    return song_df

# Load or create the interaction matrix
def load_interaction_matrix():
    try:
        interaction_matrix = pd.read_csv('data/user_interactions.csv')
    except FileNotFoundError:
        # Generate synthetic data if not found
        sp = authenticate_spotify()
        song_df = generate_synthetic_data(sp)
        interaction_matrix = pd.pivot_table(song_df, index='user_id', columns='song_id', values='play_count', fill_value=0)
        interaction_matrix.to_csv('data/user_interactions.csv', index=False)
    return interaction_matrix

# Train the SVD model
def train_recommender_model(interaction_matrix):
    reader = Reader(rating_scale=(1, interaction_matrix.max().max()))
    data = Dataset.load_from_df(interaction_matrix.stack().reset_index(), reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    model = SVD()
    model.fit(trainset)
    
    joblib.dump(model, 'models/recommender_model.pkl')
    return model

# Recommend songs for a user
def recommend_songs(user_id, model, interaction_matrix, num_recommendations=10):
    user_interactions = interaction_matrix.loc[user_id]
    user_unrated_songs = user_interactions[user_interactions == 0].index.tolist()
    
    predictions = [model.predict(user_id, song_id) for song_id in user_unrated_songs]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
    
    recommended_song_ids = [rec.iid for rec in recommendations]
    recommended_songs = interaction_matrix.columns[recommended_song_ids].tolist()
    
    return recommended_songs

# Main function to execute the recommendation
if __name__ == "__main__":
    # Authenticate and load interaction matrix
    interaction_matrix = load_interaction_matrix()
    
    # Train or load the recommender model
    try:
        model = joblib.load('models/recommender_model.pkl')
    except FileNotFoundError:
        model = train_recommender_model(interaction_matrix)
    
    # Input user ID for recommendation
    user_id = 'user_1'  # Replace with actual user ID
    recommendations = recommend_songs(user_id, model, interaction_matrix)
    
    print(f"Recommended songs for user {user_id}: {recommendations}")
