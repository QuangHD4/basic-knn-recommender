import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


class UserKNNBasicRecommender:

    def __init__(self, n_neighbors=20, metric="cosine", min_support:int=5, center_ratings=True):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.center_ratings = center_ratings
        self.min_support = min_support                    # New: min neighbors who must have rated an item

        # internal data
        self.user_ids = None
        self.item_ids = None

        self.user_index = None
        self.item_index = None

        self.user_item = None
        self.user_means = None

        self.neighbor_indices = None
        self.neighbor_sims = None


    def fit(self, ratings_df):
        # create user-item matrix
        user_item_df = ratings_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        )

        self.user_ids = user_item_df.index.to_numpy()
        self.item_ids = user_item_df.columns.to_numpy()

        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {i: j for j, i in enumerate(self.item_ids)}

        self.user_item = user_item_df.values.astype(float)

        # compute user means
        self.user_means = np.nanmean(self.user_item, axis=1)

        # center ratings if needed
        if self.center_ratings:
            centered = self.user_item - self.user_means[:, None]
        else:
            centered = self.user_item.copy()
        centered = np.nan_to_num(centered)

        sparse_matrix = csr_matrix(centered)

        # KNN model
        model = NearestNeighbors(
            metric=self.metric,
            algorithm="brute",
            n_neighbors=self.n_neighbors + 1
        )
        model.fit(sparse_matrix)
        distances, neighbors = model.kneighbors(sparse_matrix)

        # remove self neighbor
        self.neighbor_indices = neighbors[:, 1:]
        self.neighbor_sims = 1 - distances[:, 1:]
        
        sims = self.neighbor_sims.flatten()

        print("mean similarity:", sims.mean())
        print("median similarity:", np.median(sims))
        print("max similarity:", sims.max())
        print("min similarity:", sims.min())


    def predict_rating(self, user_id, item_id):

        if user_id not in self.user_index or item_id not in self.item_index:
            return np.nan

        u = self.user_index[user_id]
        i = self.item_index[item_id]

        neighbors = self.neighbor_indices[u]
        sims = self.neighbor_sims[u]

        neighbor_ratings = self.user_item[neighbors, i]

        mask = ~np.isnan(neighbor_ratings)

        if mask.sum() == 0:
            return self.user_means[u]

        neighbor_ratings = neighbor_ratings[mask]
        sims = sims[mask]
        neighbor_means = self.user_means[neighbors][mask]

        numerator = np.sum(sims * (neighbor_ratings - neighbor_means))
        denominator = np.sum(np.abs(sims))

        if denominator == 0:
            return self.user_means[u]

        return self.user_means[u] + numerator / denominator


    def predict_ratings(self, test_df):

        predictions = []

        for row in test_df.itertuples(index=False):
            predictions.append(
                self.predict_rating(row.user_id, row.item_id)
            )

        return np.array(predictions)


    def recommend_items(self, user_id, n=10):
        if user_id not in self.user_index:
            return []

        u = self.user_index[user_id]

        neighbors = self.neighbor_indices[u]
        sims = self.neighbor_sims[u]

        neighbor_ratings = self.user_item[neighbors]
        neighbor_means = self.user_means[neighbors][:, None]

        centered = neighbor_ratings - neighbor_means    # V x I; (r_{v,i} -\bar{r}_v) for all users v, items i

        # mask where neighbor actually rated the item
        mask = ~np.isnan(neighbor_ratings)

        # numerator
        weighted = sims[:, None] * centered
        weighted[~mask] = 0

        numerator = np.sum(weighted, axis=0)            # I; sum all users rating for every item

        # denominator
        denom = np.sum(np.abs(sims[:, None]) * mask, axis=0)

        # score calculation with divide-by-zero protection
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = self.user_means[u] + (numerator / denom)
        
        # also store how many neighbors rated each item for filtering
        neighbor_counts = np.sum(mask, axis=0)
        
        # handle items with no ratings from neighbors or insufficient support
        scores[denom == 0] = np.nan
        scores[neighbor_counts < self.min_support] = np.nan
        
        # clip scores to valid range
        scores = np.clip(scores, 1, 5)

        # remove items already seen by the user
        seen_mask = ~np.isnan(self.user_item[u])
        scores[seen_mask] = np.nan

        # get top items (replace NaNs with -inf for proper descending sort)
        scores_for_sort = np.nan_to_num(scores, nan=-1e9)
        top_indices = np.argsort(scores_for_sort)[::-1]

        recommendations = []
        for i in top_indices:
            if np.isnan(scores[i]):
                continue
            recommendations.append((self.item_ids[i], scores[i]))
            if len(recommendations) == n:
                break

        return recommendations
    

    def get_neighbors(self, user_id):
        """
        Return the k most similar users to the given user.

        Returns
        -------
        list of tuples:
            [(neighbor_user_id, similarity), ...]
        """

        if user_id not in self.user_index:
            return []

        u = self.user_index[user_id]

        neighbor_idxs = self.neighbor_indices[u]
        sims = self.neighbor_sims[u]

        neighbor_users = self.user_ids[neighbor_idxs]

        return list(zip(neighbor_users, sims))