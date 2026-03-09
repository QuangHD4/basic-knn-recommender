import numpy as np
import pandas as pd

class ItemMeanRecommender:
    def fit(self, df: pd.DataFrame):
        self.item_means = (
            df.groupby('item_id')['rating']
            .mean()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        self.most_popular_items = (
            df.groupby('item_id')
            .agg(mean_rating=("rating", "mean"), vote_count=("rating", "count"))
            .assign(popularity=lambda x: x["mean_rating"] * np.log1p(x["vote_count"]))
            .sort_values("popularity", ascending=False)
            .index
            .tolist()
        )


    def predict_ratings(self, user_id: int, n: int = 10):
        """
        Parameters:
        =============================================
        user_id: always ignored, exist for API consistency
        n: number of items to recommend
        """
        return self.item_means
    

    def recommend_items(self, user_id: int, n: int = 10):
        """
        Recommend top-N most popular items using item mean rating, \
            weighted by the number of votes to account for noisy ratings \
            (high rating, few votes).

        Parameters:
        =============================================
        user_id: always ignored, exist for API consistency
        n: number of items to recommend
        """
        return self.most_popular_items[:n]


class BiasBaselineRecommender:
    """
    Baseline predictor:
        r_hat_ui = global_mean + user_bias + item_bias

    Supports:
        - rating prediction (for MAE/RMSE)
        - top-N recommendation (for precision/recall)

    This accounts for:
        - User bias (user generosity): some users tend to rate higher/lower than average
        - Item bias (item likability): some items tend to receive higher/lower ratings than average
    Regularization is applied to both biases to prevent overfitting, especially for users/items with few ratings.
    """

    def __init__(self, reg: float = 10.0):
        self.reg = reg


    def fit(self, df: pd.DataFrame):
        """
        df columns: user_id, item_id, rating
        """

        self.global_mean = df["rating"].mean()

        # --- Item bias ---
        item_stats = df.groupby("item_id")["rating"].agg(["sum", "count"])
        self.item_bias = (
            (item_stats["sum"] - item_stats["count"] * self.global_mean)
            / (item_stats["count"] + self.reg)
        )

        # --- User bias ---
        df = df.join(self.item_bias.rename("item_bias"), on="item_id")

        user_stats = df.groupby("user_id").apply(
            lambda x: (x["rating"] - self.global_mean - x["item_bias"]).sum()
        )

        user_count = df.groupby("user_id").size()

        self.user_bias = user_stats / (user_count + self.reg)

        # store all items for recommendation
        self.all_items = df["item_id"].unique()


    def predict_ratings(self, test_df: pd.DataFrame):
        """
        Predict ratings for each row in test_df.

        Parameters
        ----------
        test_df : pd.DataFrame
            Columns: ["user_id", "item_id"]

        Returns
        -------
        np.ndarray
            Predicted ratings aligned with test_df rows
        """

        bu = test_df["user_id"].map(self.user_bias).fillna(0.0)
        bi = test_df["item_id"].map(self.item_bias).fillna(0.0)

        preds = self.global_mean + bu + bi

        return preds.values


    def recommend_items(self, user_id: int, n: int = 10, seen_items=None):
        """
        Rank items by predicted rating.
        """

        if seen_items is None:
            seen_items = set()

        candidates = [i for i in self.all_items if i not in seen_items]

        scores = self.predict_ratings(user_id, candidates)

        top_idx = np.argsort(scores)[::-1][:n]

        return [candidates[i] for i in top_idx]
    