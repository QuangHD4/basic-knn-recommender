import pprint
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class RecommenderEvaluator:

    def __init__(self, model, test_df, relevance_threshold=4, prec_rec_k=10):
        """
        model: trained recommender with recommend_items()
        test_df: dataframe containing user_id, item_id, rating
        relevance_threshold: rating considered 'relevant'
        prec_rec_k: k value for precision and recall calculations
        """

        self.model = model
        self.test_df = test_df
        self.relevance_threshold = relevance_threshold
        self.prec_rec_k = prec_rec_k

        # cache test users
        self.users = test_df.user_id.unique()

        # precompute relevant items per user
        self.relevant_items = (
            test_df[test_df.rating >= relevance_threshold]
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )
        # print(
        #     (test_df[test_df.rating >= relevance_threshold]
        #     .groupby("user_id")["item_id"].nunique())
        # )
        # pprint.pprint(self.relevant_items)


    def precision_recall_at_k(self):

        precisions = []
        recalls = []

        for user in self.users:

            if user not in self.relevant_items:
                continue

            relevant = self.relevant_items[user]

            recs = self.model.recommend_items(user, n=self.prec_rec_k)
            assert len(recs) == self.prec_rec_k, f"Model didn't returned k recommendations. Got {len(recs)} recommendations."
            rec_items = {item for item, _ in recs}

            hits = rec_items & relevant

            precision = len(hits) / self.prec_rec_k
            # print(len(hits), self.prec_rec_k)
            recall = len(hits) / len(relevant)

            precisions.append(precision)
            recalls.append(recall)

        return np.mean(precisions), np.mean(recalls)
    

    def ratings_error(self, err_func=[mean_absolute_error, root_mean_squared_error]):
        ratings_pred = self.model.predict_ratings(self.test_df)
        ratings_true = self.test_df['rating'].values

        coverage = np.mean(~np.isnan(ratings_pred))

        metrics = {func.__name__: func(ratings_true[~np.isnan(ratings_pred)], ratings_pred[~np.isnan(ratings_pred)]) for func in err_func}
        metrics['coverage'] = coverage

        return metrics


    def evaluation_report(self, k=10, err_func=[mean_absolute_error, root_mean_squared_error]):
        precision, recall = self.precision_recall_at_k()
        error_metrics = self.ratings_error(err_func)

        report = {
            'precision@{}'.format(k): precision,
            'recall@{}'.format(k): recall,
            **error_metrics
        }

        pprint.pprint(report)
        
        return report