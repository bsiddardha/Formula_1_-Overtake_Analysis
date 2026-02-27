import fastf1
import pandas as pd

# Enable cache (important)
fastf1.Cache.enable_cache("cache")

# Load Bahrain 2021 Race
session = fastf1.get_session(2021, "Bahrain", "R")
session.load(weather=True, laps=True)

# Copy laps
laps = session.laps.copy()

# Keep only accurate laps
laps = laps[laps['IsAccurate'] == True]
laps = laps[~laps['LapTime'].isna()]

# Convert lap time to seconds (target variable)
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Add circuit + year info
laps['Year'] = 2021
laps['Circuit'] = "Bahrain"
laps['SessionType'] = "Race"

# Select important features
df = laps[[
    'Year',
    'Circuit',
    'SessionType',
    'Driver',
    'Team',
    'LapNumber',
    'Stint',
    'Compound',
    'TyreLife',
    'FreshTyre',
    'TrackStatus',
    'Position',
    'LapTimeSeconds'
]]

print("Dataset shape:", df.shape)
print(df.head())

# Save file
df.to_csv("bahrain_2021_race_laps.csv", index=False)