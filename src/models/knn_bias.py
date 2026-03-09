import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class KNNBaselineRecommender:
    """
    KNNBaseline recommender (user-based).
    Methods:
      - fit(df): df must have columns ['user', 'item', 'rating']
      - predict_ratings(test_df): test_df same format (user,item) to predict ratings
      - recommend_items(user_id, n): return top-n item ids for user_id
    Parameters:
      - k: number of neighbors
      - reg: regularization for baseline bias estimation
      - n_iters: iterations to fit biases
      - min_common: minimum number of co-rated items to consider similarity meaningful
    """
    def __init__(self, k=20, reg=10.0, n_iters=10, min_common=1):
        self.k = k
        self.reg = reg
        self.n_iters = n_iters
        self.min_common = min_common
        # placeholders
        self.mu = None
        self.bu = None
        self.bi = None
        self.user_index = None
        self.item_index = None
        self.user_item_matrix = None
        self.user_sim = None
        self.ratings_df = None


    def fit(self, df):
        """
        Fit baseline biases and precompute user similarities.
        df: pandas DataFrame with columns ['user', 'item', 'rating']
        """
        # store original ratings
        self.ratings_df = df.copy().reset_index(drop=True)

        # create index mappings
        users = df['user_id'].unique()
        items = df['item_id'].unique()
        self.user_index = {u: idx for idx, u in enumerate(users)}
        self.item_index = {i: idx for idx, i in enumerate(items)}
        inv_user_index = {v: k for k, v in self.user_index.items()}
        inv_item_index = {v: k for k, v in self.item_index.items()}

        n_users = len(users)
        n_items = len(items)

        # global mean
        self.mu = df['rating'].mean()

        # initialize biases
        bu = np.zeros(n_users)
        bi = np.zeros(n_items)

        # build rating lists per user/item for iterative bias updates
        user_ratings = {u: [] for u in users}
        item_ratings = {i: [] for i in items}
        for _, row in df.iterrows():
            user_ratings[row['user_id']].append((row['item_id'], row['rating']))
            item_ratings[row['item_id']].append((row['user_id'], row['rating']))

        # iterative update for biases (alternating)
        for _ in range(self.n_iters):
            # update item biases
            for i in items:
                idx_i = self.item_index[i]
                num = 0.0
                den = 0.0
                for u, r in item_ratings[i]:
                    idx_u = self.user_index[u]
                    num += (r - self.mu - bu[idx_u])
                    den += 1.0
                bi[idx_i] = num / (self.reg + den) if den > 0 else 0.0

            # update user biases
            for u in users:
                idx_u = self.user_index[u]
                num = 0.0
                den = 0.0
                for i, r in user_ratings[u]:
                    idx_i = self.item_index[i]
                    num += (r - self.mu - bi[idx_i])
                    den += 1.0
                bu[idx_u] = num / (self.reg + den) if den > 0 else 0.0

        self.bu = bu
        self.bi = bi

        # build user-item matrix of baseline-adjusted ratings (NaN where missing)
        mat = np.full((n_users, n_items), np.nan, dtype=np.float32)
        for _, row in df.iterrows():
            u_idx = self.user_index[row['user_id']]
            i_idx = self.item_index[row['item_id']]
            baseline = self.mu + self.bu[u_idx] + self.bi[i_idx]
            mat[u_idx, i_idx] = row['rating'] - baseline

        self.user_item_matrix = mat

        # compute user-user similarity using cosine on rows where both have ratings
        # To handle sparsity, replace NaN with 0 for similarity but mask pairs with too few co-ratings
        mat_zero = np.nan_to_num(mat, nan=0.0)
        sim = cosine_similarity(mat_zero)
        # mask similarities where co-rated count < min_common
        # compute co-rated counts matrix
        rated_mask = ~np.isnan(mat)
        co_counts = rated_mask.astype(int) @ rated_mask.astype(int).T
        sim[co_counts < self.min_common] = 0.0
        # set diagonal to 0 to avoid self-neighboring
        np.fill_diagonal(sim, 0.0)
        self.user_sim = sim


    def _predict_single(self, user, item):
        """
        Predict a single rating for (user, item).
        If user or item unseen, fall back to reasonable defaults.
        """
        # unseen user
        if user not in self.user_index:
            # if item known, return baseline for item else global mean
            if item in self.item_index:
                i_idx = self.item_index[item]
                return self.mu + self.bi[i_idx]
            return self.mu

        u_idx = self.user_index[user]

        # unseen item
        if item not in self.item_index:
            return self.mu + self.bu[u_idx]

        i_idx = self.item_index[item]
        baseline = self.mu + self.bu[u_idx] + self.bi[i_idx]

        # neighbors who rated item
        # find users v with rating for item
        col = self.user_item_matrix[:, i_idx]
        rated_by = np.where(~np.isnan(col))[0]
        if rated_by.size == 0:
            return baseline

        sims = self.user_sim[u_idx, rated_by]
        # if all sims zero, return baseline
        if np.all(sims == 0):
            return baseline

        # pick top-k neighbors by similarity
        top_k_idx = np.argsort(-np.abs(sims))[: self.k]
        neighbors = rated_by[top_k_idx]
        sims_top = sims[top_k_idx]
        devs = self.user_item_matrix[neighbors, i_idx]  # these are r_v,i - baseline_v,i

        denom = np.sum(np.abs(sims_top))
        if denom == 0:
            return baseline

        pred = baseline + np.dot(sims_top, devs) / denom
        return pred


    def predict_ratings(self, test_df):
        """
        Predict ratings for each row in test_df (columns ['user','item'] or ['user','item','rating']).
        Returns a DataFrame with columns ['user','item','prediction'] (and 'rating' if present).
        """
        out_rows = []
        for _, row in test_df.iterrows():
            u = row['user_id']
            i = row['item_id']
            pred = self._predict_single(u, i)
            out = {'user_id': u, 'item_id': i, 'prediction': pred}
            if 'rating' in row.index:
                out['rating'] = row['rating']
            out_rows.append(out)
        return pd.DataFrame(out_rows)


    def recommend_items(self, user_id, n=10):
        """
        Recommend top-n items for user_id (items the user has not rated).
        Returns list of (item_id, predicted_rating) sorted by predicted_rating desc.
        """
        # if user unseen, recommend top-n items by global baseline (mu + bi)
        if user_id not in self.user_index:
            # recommend items with highest baseline
            items = list(self.item_index.keys())
            scores = []
            for it in items:
                i_idx = self.item_index[it]
                scores.append((it, self.mu + self.bi[i_idx]))
            scores.sort(key=lambda x: -x[1])
            return scores[:n]

        u_idx = self.user_index[user_id]
        # items user has rated
        rated_mask = ~np.isnan(self.user_item_matrix[u_idx, :])
        candidates = [it for it, idx in self.item_index.items() if not rated_mask[idx]]
        preds = []
        for it in candidates:
            pred = self._predict_single(user_id, it)
            preds.append((it, pred))
        preds.sort(key=lambda x: -x[1])
        return preds[:n]


# -------------------------
# Usage example
# -------------------------
if __name__ == "__main__":
    # small example dataset
    df = pd.DataFrame([
        {'user_id': 'A', 'item_id': 'i1', 'rating': 5.0},
        {'user_id': 'A', 'item_id': 'i2', 'rating': 3.0},
        {'user_id': 'B', 'item_id': 'i1', 'rating': 4.0},
        {'user_id': 'B', 'item_id': 'i3', 'rating': 2.0},
        {'user_id': 'C', 'item_id': 'i2', 'rating': 4.0},
        {'user_id': 'C', 'item_id': 'i3', 'rating': 5.0},
    ])

    model = KNNBaselineRecommender(k=2, reg=5.0, n_iters=10, min_common=1)
    model.fit(df)

    # predict some ratings
    test = pd.DataFrame([{'user_id': 'A', 'item_id': 'i3'}, {'user_id': 'B', 'item_id': 'i2'}])
    preds = model.predict_ratings(test)
    print(preds)

    # recommend for user A
    recs = model.recommend_items('A', n=5)
    print("Recommendations for A:", recs)