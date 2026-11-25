
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "knn_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "audio_features.pkl"

# Check required files
required_files = [
    DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH
]

for file_path in required_files:
    if not file_path.exists():
        st.error(f"Required file not found: {file_path}")
        st.stop()

# Load saved files
df = pd.read_csv(DATA_PATH)

knn_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
audio_features = joblib.load(FEATURES_PATH)

# Prepare features in exactly the same row order
X = df[audio_features].copy()
X_scaled = scaler.transform(X)

# Application title
st.title("🎵 Spotify Music Recommendation System")

st.write(
    "Discover songs similar to your favourite music "
    "using Machine Learning."
)

# Song selection
song_name = st.selectbox(
    "Select Your Favourite Song",
    sorted(df["track_name"].dropna().unique())
)

# Number of recommendations
top_n = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

# Recommendation function
def recommend_songs(song_name, top_n):

    matches = np.where(
        df["track_name"].str.lower().values
        == song_name.lower()
    )[0]

    if len(matches) == 0:
        return pd.DataFrame()

    song_index = matches[0]

    distances, indices = knn_model.kneighbors(
        X_scaled[song_index].reshape(1, -1),
        n_neighbors=top_n + 1
    )

    recommended_indices = indices.flatten()[1:]

    scores = 1 - distances.flatten()[1:]

    result = df.iloc[recommended_indices][
        ["track_name", "artists", "track_genre"]
    ].copy()

    result["Similarity Score"] = scores

    return result.reset_index(drop=True)


# Recommendation button
if st.button("Recommend Songs"):

    recommendations = recommend_songs(
        song_name,
        top_n
    )

    st.subheader("Recommended Songs")

    if recommendations.empty:
        st.warning("No recommendations found.")

    else:
        st.dataframe(
            recommendations,
            use_container_width=True
        )

        st.success(
            "Recommendations generated successfully!"
        )
