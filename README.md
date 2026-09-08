# Spotify Running ML

A machine learning pipeline that creates a personalized running playlist based on active, subjective feedback, retrains itself automatically, and pushes updates straight to Spotify.

## How it works

1. **Feedback collection** — A swipe interface (built with Lovable) shows tracks from a live Spotify playlist. Swiping records a like / dislike / maybe judgment straight into a Google Sheet.
2. **Feature data** — Audio features (danceability, energy, tempo, valence, etc.) for a track candidate pool of songs are sourced from CSV exports (since Spotify deprecated public access to its `/audio-features` endpoint for new apps in late 2024).
3. **Training** — A LightGBM regressor is trained on the swipe feedback (mapped to a 0 / 0.5 / 1 scale), using audio features and playlist position as inputs.
4. **Playlist generation** — The model scores the full candidate pool sequentially, position by position, to build a ranked top-50 playlist treating playlist position itself as a valid feature rather than a source of leakage, since it's known at each step of construction.
5. **Publishing** — The new top 50 is written back to the same Spotify playlist via the Spotify Web API (OAuth), which the swipe app then reads from, closing the loop.

## Automation
A GitHub Actions workflow runs the full pipeline (`train.py`) daily. It checks for new feedback since the last run and skips retraining if nothing new has come in, so the model only updates when there's something to learn from.
