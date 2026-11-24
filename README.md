\# Spotify Music Recommendation System



An intelligent music recommendation system that recommends similar songs using Machine Learning and Spotify audio features.



\## Project Overview



This project uses audio characteristics of songs to identify and recommend tracks with similar musical patterns. It uses the K-Nearest Neighbors (KNN) algorithm with cosine distance and provides an interactive Streamlit application.



\## Features



\- Search and select songs from the dataset

\- Recommend similar songs

\- Choose the number of recommendations

\- Compare songs using audio features

\- Interactive Streamlit interface



\## Technology Stack



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- K-Nearest Neighbors (KNN)

\- Cosine Distance

\- StandardScaler

\- Streamlit

\- Joblib

\- Jupyter Notebook



\## Machine Learning Workflow



1\. Data loading and exploration

2\. Data cleaning and preprocessing

3\. Duplicate track removal

4\. Audio feature selection

5\. Feature scaling using StandardScaler

6\. KNN model training

7\. Similarity-based song recommendations

8\. Streamlit application development



\## Audio Features Used



\- Danceability

\- Energy

\- Valence

\- Acousticness

\- Instrumentalness

\- Tempo

\- Loudness



\## Project Structure



```text

Spotify\_Music\_Recommendation/

├── app/

│   └── app.py

├── notebooks/

│   └── Spotify\_Music\_Recommendation.ipynb

├── dataset/

│   └── spotify\_cleaned.csv

├── models/

│   ├── knn\_model.pkl

│   ├── scaler.pkl

│   └── audio\_features.pkl

├── screenshots/

├── docs/

├── README.md

├── requirements.txt

└── .gitignore

```



\## Installation and Execution



Clone the repository:



```bash

git clone https://github.com/YOUR\_USERNAME/spotify-music-recommendation-system.git

cd spotify-music-recommendation-system

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Make sure the cleaned dataset and trained model files are available in their respective folders.



Run the application:



```bash

python -m streamlit run app/app.py

```



\## Dataset



Source: \[Spotify Tracks Dataset – Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)



Please refer to the original dataset's licence and terms before redistributing the data.



\## Future Improvements



\- Mood-based recommendations

\- Genre-based filtering

\- Personalized playlists

\- Spotify API integration

\- Improved recommendation evaluation



\## Author



LOKESH



GitHub: https://github.com/Lokesh-lLoki/spotify-music-recommendation-system.git



\## Disclaimer



This is an educational Machine Learning portfolio project. It is not an official Spotify product.

