import streamlit as st
import pandas as pd
from content_filter import recommend_movies

# Load movies dataset
movies = pd.read_csv("movies.csv")

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Find your next favorite movie!")

# Movie selection
movie_title = st.selectbox("Select a movie:", movies["title"].values)

if st.button("Get Recommendations"):
    recommendations = recommend_movies(movie_title)
    st.write("✅ Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
