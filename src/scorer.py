from .evaluator import RecommenderEvaluator

class RecommenderScorer:
    """Scorer that evaluates recommender models using RecommenderEvaluator."""
    def __call__(self, estimator, X_test, y_test=None):
        # X_test is the test DataFrame (already split by PredefinedSplit)
        # estimator is already trained on the training fold
        evaluator = RecommenderEvaluator(estimator, X_test, relevance_threshold=4, prec_rec_k=10)
        results = evaluator.evaluation_report(k=10)
        return results.get('precision@10', 0)