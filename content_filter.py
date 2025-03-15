import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv("movies.csv")

# Generate a 'description' column (placeholder using genres)
movies["description"] = movies["genres"].fillna("")

# Convert text descriptions into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["description"])

# Compute cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on movie title
def recommend_movies(title, num_recommendations=5):
    if title not in movies["title"].values:
        return f"❌ Movie '{title}' not found in dataset."

    # Get the movie index
    idx = movies[movies["title"] == title].index[0]

    # Get similarity scores & sort them
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    # Get recommended movie titles
    recommended_movies = [movies.iloc[i[0]]["title"] for i in scores]
    return recommended_movies

# Example usage
print("✅ Content-Based Recommendations for 'Toy Story (1995)':")
print(recommend_movies("Toy Story (1995)"))
