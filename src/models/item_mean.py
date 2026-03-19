import numpy as np
import pandas as pd

class ItemMeanRecommender:
    @property
    def __name__(self):
        return type(self).__name__


    def fit(self, df: pd.DataFrame):
        self.trained_users = df["user_id"].unique()
        self.item_means = (
            df.groupby('item_id')['rating']
            .mean()
            .sort_values(ascending=False)
            .to_dict()
        )
        self.global_mean = df["rating"].mean()

        self.seen_items = df.groupby("user_id")["item_id"].apply(set).to_dict()

        self.most_popular_items = (
            df.groupby('item_id')
            .agg(mean_rating=("rating", "mean"), vote_count=("rating", "count"))
            .assign(popularity=lambda x: x["mean_rating"] * np.log1p(x["vote_count"]))
            .sort_values("popularity", ascending=False)
            .index
            .tolist()
        )


    def predict_ratings(self, test_df: pd.DataFrame):
        return test_df["item_id"].apply(lambda x: self.item_means.get(x, self.global_mean)).values
    

    def recommend_items(self, user_id: int, n: int = 10):
        """
        Recommend top-N most popular items using item mean rating, \
            weighted by the number of votes to account for noisy ratings \
            (high rating, few votes).

        Parameters:
        =============================================
        user_id: user to filter recommendations for (new items only)
        n: number of items to recommend
        """
        if user_id not in self.trained_users:
            return []
            
        seen = self.seen_items.get(user_id, set())
        
        recs = []
        for item in self.most_popular_items:
            if item not in seen:
                recs.append(item)
            if len(recs) == n:
                break
        return recs
