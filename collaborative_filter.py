import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample ratings dataset
ratings_dict = {
    "userId": [1, 2, 3, 4, 5],
    "movieId": [1, 2, 1, 3, 4],
    "rating": [5, 3, 4, 5, 2]
}
ratings = pd.DataFrame(ratings_dict)

# Convert ratings to a pivot table (user-movie matrix)
pivot_table = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Compute similarity between movies
movie_similarity = cosine_similarity(pivot_table.T)

# Function to recommend movies based on ratings
def recommend_by_ratings(movie_id, num_recommendations=5):
    if movie_id not in pivot_table.columns:
        return f"❌ Movie ID {movie_id} not found."

    idx = list(pivot_table.columns).index(movie_id)
    scores = list(enumerate(movie_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    recommended_movie_ids = [pivot_table.columns[i[0]] for i in scores]
    return recommended_movie_ids

# Example usage
print("✅ Collaborative Filtering Recommendations for Movie ID 1:")
print(recommend_by_ratings(1))
