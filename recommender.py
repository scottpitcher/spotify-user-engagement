import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from surprise import Dataset, Reader, SVD
from sklearn.model_selection import train_test_split
import joblib
import dotenv
import os
import time


dotenv.load_dotenv()
# Load Spotify API credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv('SPOTIPY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
))

playlists = {
        "Today's Top Hits": "37i9dQZF1DXcBWIGoYBM5M",
        "Global Top 50": "37i9dQZEVXbMDoHDwVN2tF",
        "Global Viral 50": "37i9dQZEVXbLiRSasKsNU9",
        "New Music Friday": "37i9dQZF1DX4JAvHpjipBk",
        "Hot Country": "37i9dQZF1DX1lVhptIYRda",
        "Beast Mode": "37i9dQZF1DX76Wlfdnj7AP",
        "Chill Hits": "37i9dQZF1DX4WYpdgoIcn6",
        "Soft Pop Hits": "37i9dQZF1DX3YSRoSdA634",
        "Good Vibes": "37i9dQZF1DX6GwdWRQMQpq",
        "Evening Acoustic": "37i9dQZF1DXbJmiEZs5p2t",
        "All Out 80s": "37i9dQZF1DX4UtSsGT1Sbe",
        "All Out 90s": "37i9dQZF1DXbTxeAdrVG2l",
        "Your Favorite Coffeehouse": "37i9dQZF1DX6ziVCJnEm59",
        "Acoustic Hits": "37i9dQZF1DX4E3UdUs7fUx",
        "Deep Focus": "37i9dQZF1DWZeKCadgRdKQ",
        "Throwback Thursday": "37i9dQZF1DX4UtSsGT1Sbe",
        "Peaceful Guitar": "37i9dQZF1DX0jgyAiPl8Af",
        "Classic Road Trip Songs": "37i9dQZF1DWSThc8QnxalT",
        "Relax & Unwind": "37i9dQZF1DX6MOzVr6s0AO",
        "Top 50 USA": "37i9dQZEVXbLRQDuF5jeBp",
        "Viral 50 USA": "37i9dQZEVXbKuaTI1Z1Afx",
        "Top 50 UK": "37i9dQZEVXbLnolsZ8PSNw",
        "Viral 50 UK": "37i9dQZEVXbL3DLHfQeDmV",
        "Top 50 Brazil": "37i9dQZEVXbMXbN3EUUhlg",
        "Viral 50 Brazil": "37i9dQZEVXbMMy2roB9myp",
        "Top 50 France": "37i9dQZEVXbIPWwFssbupI",
        "Viral 50 France": "37i9dQZEVXbIZM8SIgu6df",
        "Top 50 Japan": "37i9dQZEVXbKXQ4mDTEBXq",
        "Viral 50 Japan": "37i9dQZEVXbKqiTGXuCOsB",
        "Top 50 India": "37i9dQZEVXbLZ52XmnySJg",
        "Viral 50 India": "37i9dQZEVXbMWDif5SCBJq",
        "Top 50 Italy": "37i9dQZEVXbIQnj7RRhdSX",
        "Viral 50 Italy": "37i9dQZEVXbKbvcwe5owJ1",
        "Top 50 South Korea": "37i9dQZEVXbJZyENOWUFo7",
        "Viral 50 South Korea": "37i9dQZEVXbNxXF4SkHj9F",
        "Top 50 Australia": "37i9dQZEVXbJPcfkRz0wJ0",
        "Viral 50 Australia": "37i9dQZEVXbK4fwx2r07XW",
        "Top 50 Germany": "37i9dQZEVXbJiZcmkrIHGU",
        "Viral 50 Germany": "37i9dQZEVXbKglSdDwFtE9"
    }
def get_tracks(playlist_id):
    try:
        results = sp.playlist_tracks(playlist_id)
        tracks = results['items']
        
        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])
        return tracks
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error fetching playlist {playlist_id}: {e}")
        return []

def generate_playlist_data():
    """getting """
    song_data = []
    playlist_data = []
    for playlist_name, playlist_id in playlists.items():
        tracks = get_tracks(playlist_id)
        for song in tracks:
            track = song.get('track')
            if track is not None:  # Check if track is not None
                song_data.append({
                    'song_id': track.get('id'),
                    'title': track.get('name'),
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'duration': track['duration_ms'] // 1000,  # convert ms to seconds
                    'popularity': track['popularity'],
                    'release_date': track['album']['release_date']
                })

                playlist_data.append({
                    'playlist_name': playlist_name,
                    'song_id': track.get('id'),
                })
        
        # Respect API limit
        time.sleep(2)
    
    song_df = pd.DataFrame(song_data)
    playlist_df = pd.DataFrame(playlist_data)
    return song_df, playlist_df

def train_recommender_model(interaction_data):
    """Train the SVD model"""
    reader = Reader(rating_scale=(1, interaction_data['play_count'].max()))
    data = Dataset.load_from_df(interaction_data[['user_id', 'song_id', 'play_count']], reader)
    trainset = data.build_full_trainset()
    
    model = SVD()
    model.fit(trainset)
    joblib.dump(model, 'models/recommender_model.pkl')
    return model


def recommend_playlist(user_id, model, interaction_data, playlist_df, num_recommendations=1):
    """Recommend a playlist for a user based on their listening history"""
    if user_id not in interaction_data['user_id'].unique():
        raise ValueError(f"User ID {user_id} does not exist in the interaction data.")
    
    user_interactions = interaction_data[interaction_data['user_id'] == user_id]
    user_rated_songs = user_interactions['song_id'].tolist()

    # predict ratings for songs the user hasn't rated
    all_songs = interaction_data['song_id'].unique()
    user_unrated_songs = [song for song in all_songs if song not in user_rated_songs]

    # predict ratings for unrated songs
    predictions = [model.predict(user_id, song_id) for song_id in user_unrated_songs]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)

    # create a dataframe with recommended songs and their estimated ratings
    recommended_songs = pd.DataFrame({
        'song_id': [rec.iid for rec in recommendations],
        'estimated_rating': [rec.est for rec in recommendations]
    })

    playlist_recommendations = pd.merge(recommended_songs, playlist_df, on='song_id')
    # Normalize the estimated ratings [0,5]
    min_rating = recommended_songs['estimated_rating'].min()
    max_rating = recommended_songs['estimated_rating'].max()
    recommended_songs['normalized_rating'] = (recommended_songs['estimated_rating'] - min_rating) / (max_rating - min_rating) * 4 + 1

    # Aggregate the estimated ratings per playlist
    playlist_scores = playlist_recommendations.groupby('playlist_name')['normalized_rating'].sum().reset_index()

    # Sort playlists by their scores, recommend num_recommendations
    top_playlists = playlist_scores.sort_values(by='normalized_rating', ascending=False).head(num_recommendations)

    return top_playlists

# Executing the function
if __name__ == "__main__":
    # Loading in the user-interaction dataframe
    interaction_data = pd.read_csv('data/user/user_data.csv') 

    # train or load the recommender model
    try:
        model = joblib.load('models/recommender_model.pkl')
    except FileNotFoundError:
        model = train_recommender_model(interaction_data)
    
    # Generate playlist data to get playlist information
    _, playlist_df = generate_playlist_data()
    
    # Input user ID for recommendation
    user_id = 'usa_user_1'
    recommendations = recommend_playlist(user_id, model, interaction_data, playlist_df)
    print(f"Recommended playlists for user {user_id}:\n", recommendations)