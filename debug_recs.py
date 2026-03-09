import pandas as pd
import numpy as np

from src.models import UserKNNBasicRecommender
from src.data_splits import load_premade_splits

# Load only fold 1 for speed
folds = list(load_premade_splits("data"))
train_df, test_df = folds[0]

model_knn = UserKNNBasicRecommender(n_neighbors=50, metric='cosine', min_support=10, center_ratings=False)
model_knn.fit(train_df)

# Pick a test user
test_users = test_df.user_id.unique()
user = test_users[0]

recs = model_knn.recommend_items(user, n=10)

print("Recommendations for user", user)
for item, score in recs:
    print(f"Item: {item}, Score: {score}")

# Let's also print what the actual denom was for these items
u = model_knn.user_index[user]
neighbors = model_knn.neighbor_indices[u]
sims = model_knn.neighbor_sims[u]

neighbor_ratings = model_knn.user_item[neighbors]
neighbor_means = model_knn.user_means[neighbors][:, None]
centered = neighbor_ratings - neighbor_means
mask = ~np.isnan(neighbor_ratings)

weighted = sims[:, None] * centered
weighted[~mask] = 0
numerator = np.sum(weighted, axis=0)
denom = np.sum(np.abs(sims[:, None]) * mask, axis=0)
scores = model_knn.user_means[u] + numerator / denom

print("\nDetailed stats for recommended items:")
for item, score in recs:
    i = model_knn.item_index[item]
    print(f"Item: {item}, Score: {score:.4f}, Num: {numerator[i]:.4f}, Denom: {denom[i]:.4f}, Support: {mask[:, i].sum()}")
