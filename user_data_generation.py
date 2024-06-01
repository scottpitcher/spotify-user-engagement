# Purpose: Generate user-interaction data based on real song info from Spotify API

# Importing packages
import pandas as pd

# Load in the songs data
with open("data/song_data.csv", 'r') as file:
    songs = file.read()

# Create synthetic data
## Desired columns: User ID, Song ID, Play Count, Last Played, User Age, User Country
user_data = pd.DataFrame()


with open("data/user_data.csv", 'w') as outfile:
    outfile.write(user_data)