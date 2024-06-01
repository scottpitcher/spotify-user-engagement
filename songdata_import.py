# Purpose: Utilise the Spotify Web API to generate user-interaction data based on real song info

# Importing packages
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import dotenv

# Initialising the spotify API into the environment
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=dotenv.load_dotenv('SPOTIFY_CLIENT_ID'),
    client_secret=dotenv.load_dotenv('SPOTIFY_CLIENT_SECRET')
))


