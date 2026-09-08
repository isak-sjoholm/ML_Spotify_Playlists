

# Imports & Installations

## General
import pandas as pd
import glob
import os

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
print(f"Inloggad som: {user['display_name']} ({user['id']})")

playlist_id = "0kmA3oRKWHZNrqS2zByAR3"
