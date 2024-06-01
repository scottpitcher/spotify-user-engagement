# Purpose: Utilise the Spotify Web API to generate user-interaction data based on real song info

# Importing packages
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import dotenv
import os
import time

dotenv.load_dotenv()

# Initialising the spotify API into the environment
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv('SPOTIPY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
))

def get_tracks(playlist_id):
    """Retrieving all tracks from a playlists ID"""
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def extract_info(playlists):
    """Extracting information from a song,"""
    song_data = []
    for playlist_id in playlists.values():
        tracks = get_tracks(playlist_id)

        for song in tracks:
            track = song['track']
            song_data.append({
                'song_id': track['id'],
                'title': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'duration': track['duration_ms'] // 1000,
                'popularity': track['popularity'],
                'release_date': track['album']['release_date']
            })
        # Pause to respect API rate limits
        time.sleep(1)
    return pd.DataFrame(song_data)

# Global Playlists
global_playlists = {"Today's Top Hits":"37i9dQZF1DXcBWIGoYBM5M"}

# Country-specific playlists



# 
global_df = extract_info(global_playlists)
global_df.to_csv("data/songs/global_song_data.csv", index=False)


