

# Imports & Installations

## General
import pandas as pd
import glob
import os
import json


## Google
import gspread
from google.oauth2.service_account import Credentials

## ML-related
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from scipy.stats import spearmanr

## Spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth



# Set up Spotify integration
CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')

REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPE = "playlist-modify-private playlist-modify-public"

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
)

REFRESH_TOKEN = os.environ.get('SPOTIFY_REFRESH_TOKEN')

token_info = sp_oauth.refresh_access_token(REFRESH_TOKEN)
sp = spotipy.Spotify(auth=token_info['access_token'])

user = sp.current_user()
print(f"Logged in as: {user['display_name']} ({user['id']})")

playlist_id = "0kmA3oRKWHZNrqS2zByAR3"



# Read & deduplicate Spotify data
path = 'data/'
files = glob.glob(path + '*.csv')

dfs = []

for file in files:
    try:
        df = pd.read_csv(file)
        df['källfil'] = file.split('/')[-1]
        dfs.append(df)

    except Exception as e:
        print(f"Could not read {file}: {e}")

combined = pd.concat(dfs, ignore_index=True)
print(f"Total numer of rows (inc. duplicates): {len(combined)}")

deduped = combined.drop_duplicates(subset='Låtens URI', keep='first')
print(f"Total number of unique songs: {len(deduped)}")



# Read song feedback from Google Sheets & join with audio Spotify data

## Authenticate to Google Sheets via service account
service_account_info = json.loads(os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON'))
scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
gc = gspread.authorize(creds)

## Open & read Google Sheets with song feedbacks
sheet = gc.open("Run Music Feedback").sheet1
rows = sheet.get_all_values()

## Save feedback in df
feedback_df = pd.DataFrame(rows, columns=['created_at', 'track_uri', 'track_position', 'feedback'])
feedback_df['track_position'] = feedback_df['track_position'].astype(int)

## Join with Spotify data
training_data = feedback_df.merge(deduped, left_on='track_uri', right_on='Låtens URI', how='left')
training_data_clean = training_data[training_data['Låtens URI'].notna()].copy()



# Set up the data with correct features and target
feedback_map = {'dislike': 0.0, 'maybe': 0.5, 'like': 1.0}
training_data_clean['target'] = training_data_clean['feedback'].map(feedback_map)

feature_cols = ['Dansbarhet', 'Energi', 'Tonart', 'Ljudstyrka', 'Läge', 'Talighet', 'Akustik', 'Instrumentalhet', 'Livlighet', 'Valens', 'Tempo', 'Taktart', 'Popularitet', 'track_position']
X = training_data_clean[feature_cols].copy()
y = training_data_clean['target'].copy()



# Split into a stratified test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=training_data_clean['feedback'])



# Train a LightGBM model to forecast a numeric transformation of the feedback target Dislike/Maybe/Like
train_set = lgb.Dataset(X_train, label=y_train)
test_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

params = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'seed': 42
}

model = lgb.train(
    params,
    train_set,
    num_boost_round=200,
    valid_sets=[test_set],
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
)



# Predict target for all valid songs for each playlist position and save top 50
candidate_pool = deduped.rename(columns={
    'Dansbarhet': 'Dansbarhet', 'Energi': 'Energi', 'Tonart': 'Tonart',
    'Ljudstyrka': 'Ljudstyrka', 'Läge': 'Läge', 'Talighet': 'Talighet',
    'Akustik': 'Akustik', 'Instrumentalhet': 'Instrumentalhet',
    'Livlighet': 'Livlighet', 'Valens': 'Valens', 'Tempo': 'Tempo',
    'Taktart': 'Taktart', 'Popularitet': 'Popularitet'
}).copy()

# Remove songs with missing audio features
audio_feature_cols = [c for c in feature_cols if c != 'track_position']
candidate_pool = candidate_pool.dropna(subset=audio_feature_cols).reset_index(drop=True)
remaining_pool = candidate_pool.copy()

playlist_rows = []

# Loop over each playlist position & predict target value (necessary since playlist position is fed as input feature)
for position in range(50):
    remaining_pool['track_position'] = position
    scores = model.predict(remaining_pool[feature_cols])
    best_idx = scores.argmax()

    best_row = remaining_pool.iloc[best_idx].copy()
    best_row['predicted_score'] = scores[best_idx]
    playlist_rows.append(best_row)

    remaining_pool = remaining_pool.drop(remaining_pool.index[best_idx]).reset_index(drop=True)

playlist_df = pd.DataFrame(playlist_rows)
print(playlist_df[['Låtens namn', 'Artistens namn', 'predicted_score']])
