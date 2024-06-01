# Purpose: Utilise the Spotify Web API to generate user-interaction data based on real song info
# Notes: We will use the country-specific playlists to generate more accurate, specific user data

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

def extract_info(playlists):
    """Extracting information from a song,"""
    song_data = []
    for playlist_id in playlists.values():
        tracks = get_tracks(playlist_id)

        for song in tracks:
            track = song.get('track')
            if track is not None:  # Check if track is not None
                song_data.append({
                    'song_id': track.get('id'),
                    'title': track.get('name'),
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
# Will be given more weighting as more popular
global_hot = {"Today's Top Hits": "37i9dQZF1DXcBWIGoYBM5M",
    "Global Top 50": "37i9dQZEVXbMDoHDwVN2tF",
    "Global Viral 50": "37i9dQZEVXbLiRSasKsNU9",
    "New Music Friday": "37i9dQZF1DX4JAvHpjipBk"}

# Will be given less weighting as less popular
global_random = {"Hot Country": "37i9dQZF1DX1lVhptIYRda",
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
    "Relax & Unwind": "37i9dQZF1DX6MOzVr6s0AO"}

# Country-specific playlists: USA, UK, Brazil, France, Japan, India, Spain, South Korea, Australia, Germany
usa = {
    "Top 50 USA": "37i9dQZEVXbLRQDuF5jeBp",
    "Viral 50 USA": "37i9dQZEVXbKuaTI1Z1Afx"
}

uk= {
    "Top 50 UK": "37i9dQZEVXbLnolsZ8PSNw",
    "Viral 50 UK": "37i9dQZEVXbL3DLHfQeDmV"
}

brazil = {
    "Top 50 Brazil": "37i9dQZEVXbMXbN3EUUhlg",
    "Viral 50 Brazil": "37i9dQZEVXbMMy2roB9myp"
}

france= {
    "Top 50 France": "37i9dQZEVXbIPWwFssbupI",
    "Viral 50 France": "37i9dQZEVXbIZM8SIgu6df"
}

japan = {
    "Top 50 Japan": "37i9dQZEVXbKXQ4mDTEBXq",
    "Viral 50 Japan": "37i9dQZEVXbKqiTGXuCOsB"
}

india= {
    "Top 50 India": "37i9dQZEVXbLZ52XmnySJg",
    "Viral 50 India": "37i9dQZEVXbMWDif5SCBJq"
}

italy= {
    "Top 50 Italy": "37i9dQZEVXbIQnj7RRhdSX",
    "Viral 50 Italy": "37i9dQZEVXbKbvcwe5owJ1"
}

southkorea = {
    "Top 50 South Korea": "37i9dQZEVXbJZyENOWUFo7",
    "Viral 50 South Korea": "37i9dQZEVXbNxXF4SkHj9F"
}

australia= {
    "Top 50 Australia": "37i9dQZEVXbJPcfkRz0wJ0",
    "Viral 50 Australia": "37i9dQZEVXbK4fwx2r07XW"
}

germany= {
    "Top 50 Germany": "37i9dQZEVXbJiZcmkrIHGU",
    "Viral 50 Germany": "37i9dQZEVXbKglSdDwFtE9"
}

# Saving these all to a list to loop over
all_playlists = {
    "global_hot": global_hot,
    "global_random":global_random,
    "usa": usa,
    "uk": uk,
    "brazil": brazil,
    "france": france,
    "japan": japan,
    "india": india,
    "italy": italy,
    "southkorea": southkorea,
    "australia": australia,
    "germany": germany
}

# Saving these playlists to .csv's for later usage
for country_name, playlists in all_playlists.items():
    country_df = extract_info(playlists).drop_duplicates()
    if not country_df.empty:
        country_df.to_csv(f"data/songs/{country_name}.csv", index=False)
        print(f"Processed {country_name} successfully")
    else:
        print(f"No data for {country_name}")


