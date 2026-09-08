

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