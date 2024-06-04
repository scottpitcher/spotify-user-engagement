# 🎵 Spotify Playlist Recommender and User Engagement/Retention Analysis 📈

## 🎯 Objectives
- Build a recommendation system to suggest songs based on user preferences
- Predict user engagement levels
- Analyze factors influencing user retention and predict churn

## 📊 Data
1. Songs Data: Song ID, Title, Artist, Album, Duration (s), Popularity, Release Date

<b> Source:</b> [Spotify Web API](https://developer.spotify.com/documentation/web-api) and connecting through spotipy library in Python

2. User-Engagement Data: User ID, Song ID, Play Count, Last Played, User Age, User Country

<b> Source:</b> Synthetic data generated from Spotify API data*


**For this project, the Spotify API'd data will come from a selection of public, Spotify-created playlists; these playlists will be both Global and Country-specific. Country-specific data will be used to create more accurate synthetic user-data from those respective countries, while Global songs will be scattered through all users listening histories. The code for this process can be found in <b> *user_data_generation.py*</b> script*
