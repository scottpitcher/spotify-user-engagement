# Purpose: Generate user-interaction data based on real song info from Spotify API
# Notes: We will use the country-specific playlists to generate more accurate, specific user data

# To do this, we will loop over every country used in the playlist dataframe collections
# Then, over each iteration we will create ~ 100 users per country where each playlist will get the following weightings:
# Global Hot: 80
# Global Random: 10
# Their Country: 60
# Other Countries: 0
# The playlists' tracks will be duplicated by those weightings * song popularity
# The Global df's will be concatenated along with their respective country
# A randomizer will select 500 song ID's per user from this dataframe, with replacement
# Once 500 songs have been chosen, the df will be grouped by song id, a count column will be created, then duplicate songs will be dropped
# Thus, each person will have a listening history of 500 plays, but ranging number of songs

# This process is all to create accurate synthetic data

# Importing packages
import pandas as pd
import os
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
# Checking to make sure all dataframes were read in correctly with the correct variable name
for name, value in globals().items():
    if name.endswith('df') :
        print(name)

# Create synthetic data
## Desired columns: User ID, Song ID, Play Count, Last Played, User Age, User Country

for country in countries:
    print(country)