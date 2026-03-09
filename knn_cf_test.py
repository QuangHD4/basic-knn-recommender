import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Recommender-system-related implementations
from src.models import ItemMeanRecommender, BiasBaselineRecommender   # baselines
from src.models import UserKNNBasicRecommender                        # main model
from src.models import KNNBaselineRecommender                         # improved model

from src.data_splits import load_premade_splits
from src.evaluator import RecommenderEvaluator
from src.debug import low_coverage_check

df = pd.read_csv(
    'data/u.data', 
    sep='\t', 
    encoding='utf-8', 
    header=None, 
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

df.sort_values('timestamp', ascending=True, inplace=True)

for fold, (train_df, test_df) in enumerate(load_premade_splits("data"), start=1):

    # =================== training ===================
    model_knn = UserKNNBasicRecommender(n_neighbors=50, metric='cosine', min_support=10, center_ratings=False)
    model_knn.fit(train_df)

    knn_bias = KNNBaselineRecommender(k=50, reg=10.0, n_iters=10, min_common=5)
    knn_bias.fit(train_df)
    print('='*75, '\n', "Fold: ", fold, sep='')
    # ================= sanity check =================
    # print("sanity check:\n", '-'*25, sep='')
    # low_coverage_check(train_df, test_df, model_knn)
    # ================== evaluation ==================
    # print("\nevaluation metrics:\n", '-'*25, sep='')
    evaluator = RecommenderEvaluator(model_knn, test_df, relevance_threshold=4, prec_rec_k=10)
    eval_results = evaluator.evaluation_report()

    evaluator_knn_bias = RecommenderEvaluator(knn_bias, test_df, relevance_threshold=4, prec_rec_k=10)
    eval_results_knn_bias = evaluator_knn_bias.evaluation_report()
