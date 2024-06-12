# Purpose: Generate user-interaction data based on real song info from Spotify API
# Notes: We will use the country-specific playlists to generate more accurate, specific user data

# To do this, we will loop over every country used in the playlist dataframe collections
# Then, over each iteration we will create ~ 100 users per country where each playlist will get the following weightings:
# Global Hot: 90
# Global Random: 1
# Their Country: 60
# Other Countries: 0
# The playlists' tracks will be duplicated by those weightings * song popularity
# The Global df's will be concatenated along with their respective country
# A randomizer will select 500 song ID's per user from this dataframe, with replacement
# Once 500 songs have been chosen, the df will be grouped by song id, a count column will be created, then duplicate songs will be dropped
# Then, the play counts will be transformed to mimic real streaming habits
# Thus, each person will have a listening history of 500 plays, but ranging number of songs

# This process is all to create accurate synthetic data

# Importing packages
import pandas as pd
import os
import random
from datetime import datetime, timedelta
import numpy as np
directory = "data/songs/"

# Reading in the dataframes
country_dataframes = {}
countries = []
# Looping through data folder
for filename in os.listdir(directory):
    if filename.endswith(".csv"):

        country_name = filename[:-4]
        countries.append(country_name)

        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        country_dataframes[country_name] = df
    

        globals()[f"{country_name}_df"] = df

countries.remove('global_random')
countries.remove('global_hot')
print(countries)

del country_dataframes['global_hot']
del country_dataframes['global_random']
print(country_dataframes.keys())

# # Checking to make sure all dataframes were read in correctly with the correct variable name
# for name, value in globals().items():
#     if name.endswith('_df'):
#         print(name)

# Transform play count
def transform_play_counts(play_counts, power=1.5, scale=10, noise_scale= 5):
    """Transform fake play counts into more realistic ones using a power transformation with a scale"""
    transformed_counts = np.power(play_counts, power)
    transformed_counts = transformed_counts * scale

    # adding some random noise to simulate variability in user behavior
    noise = np.random.normal(loc=0, scale=noise_scale, size=len(transformed_counts))
    transformed_counts += noise
    
    transformed_counts = np.maximum(transformed_counts, 1) # Ensure there are no negative values; also ensure each play count has at least 1
    
    return transformed_counts.astype(int)


# Create synthetic data
def repeat_rows(df, weight):
    """Function to duplicate rows in a dataframe based on weight and popularity"""
    return df.loc[np.repeat(df.index, df['popularity'] * weight)]

def generate_user_interaction_data(country_name, country_df, global_hot_df= global_hot_df, global_random_df = global_random_df, num_users=100):
    users_data = []

    for user_id in range(num_users):
        user_age = random.randint(14, 65) # Randomly assign age
        # Repeat rows according to weights (taking user age into account for trending vs. older songs)
        global_hot_rep = repeat_rows(global_hot_df, 100-user_age)
        global_random_rep = repeat_rows(global_random_df, user_age)
        country_rep = repeat_rows(country_df, 60)

        # Combine the DataFrames
        combined_df = pd.concat([global_hot_rep, global_random_rep, country_rep])

        # Randomly select 500 songs for the user with replacement
        user_songs = combined_df.sample(n=500, replace=True)

        # Group by song_id and count plays
        user_songs = user_songs.groupby('song_id').size().reset_index(name='play_count')

        # Add user details
        user_songs['user_id'] = f'{country_name}_user_{user_id}'
        user_songs['last_played'] = [datetime.now() - timedelta(days=random.randint(0, 90)) for _ in range(len(user_songs))] # Last played as random date within 90 days
        user_songs['user_age'] = user_age
        user_songs['user_country'] = country_name
        user_songs['play_count'] = transform_play_counts(user_songs['play_count']) # transform play count using premade function

        users_data.append(user_songs)

    return pd.concat(users_data)

## Desired columns: User ID, Song ID, Play Count, Last Played, User Age, User Country
user_data = pd.DataFrame(columns=['user_id', 'song_id', 'play_count', 'last_played', 'user_age', 'user_country'])

for country in countries:
    """Looping over each country so that we can create users from each"""
    new_users = generate_user_interaction_data(country_name = country, country_df= country_dataframes[country])
    user_data = pd.concat([user_data, new_users], ignore_index=True)
    print(f"{country} completed.")

print(user_data.nunique())

user_data.to_csv('data/user/user_data.csv', index=False)