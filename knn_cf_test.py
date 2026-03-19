import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.stats import randint
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Recommender-system-related implementations
from src.models import ItemMeanRecommender              # baseline
from src.models import UserKNNBasicRecommender          # main models

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

# # ================= HYPERPARAMETER TUNING =================
# print("--- Hyperparameter Tuning ---")
# param_dist_basic = {
#     'n_neighbors': randint(5, 60),
#     'min_support': randint(2, 15),
#     'center_ratings': [True, False]
# }

# # Use the first fold for tuning to save time
# train_df_tuning, test_df_tuning = next(load_premade_splits("data"))

# # Prepare combined data and PredefinedSplit
# train_size = len(train_df_tuning)
# test_size = len(test_df_tuning)
# combined_df = pd.concat([train_df_tuning, test_df_tuning], ignore_index=True)
# split_index = [-1]*train_size + [0]*test_size
# ps = PredefinedSplit(split_index)

# # Custom scorer class for RecommenderEvaluator
# class RecommenderScorer:
#     """Scorer that evaluates recommender models using RecommenderEvaluator."""
#     def __call__(self, estimator, X_test, y_test=None):
#         # X_test is the test DataFrame (already split by PredefinedSplit)
#         # estimator is already trained on the training fold
#         evaluator = RecommenderEvaluator(estimator, X_test, relevance_threshold=4, prec_rec_k=10)
#         results = evaluator.evaluation_report(k=10)
#         return results.get('precision@10', 0)

# scorer = RecommenderScorer()

# print("Tuning UserKNNBasic...")
# rs_basic = RandomizedSearchCV(UserKNNBasicRecommender(), param_distributions=param_dist_basic, n_iter=50, scoring=scorer, cv=ps, random_state=42, refit=False, n_jobs=1)
# rs_basic.fit(combined_df, None)
# best_params_basic = rs_basic.best_params_
# print(f"Basic Best params: {best_params_basic} | Score: {rs_basic.best_score_}")

# # ================= CROSS-VALIDATION EVALUATION =================
# print("\n--- Cross-Validation Evaluation ---")

# # Collect all fold results
# cv_fold_results = []

# for fold, (train_df, test_df) in enumerate(load_premade_splits("data"), start=1):

#     # =================== training ===================
#     item_mean_baseline = ItemMeanRecommender()
#     item_mean_baseline.fit(train_df)


#     model_knn = UserKNNBasicRecommender(**best_params_basic)
#     model_knn.fit(train_df)

#     # print('='*75, '\n', "Fold: ", fold, sep='')

#     # ================= sanity check =================
#     # print("sanity check:\n", '-'*25, sep='')
#     # low_coverage_check(train_df, test_df, model_knn)

#     # ================== evaluation ==================
#     eval_report = {}
#     for model in [item_mean_baseline, model_knn]:
#         evaluator = RecommenderEvaluator(model, test_df, relevance_threshold=4, prec_rec_k=10)
#         eval_results = evaluator.evaluation_report()
#         eval_report[model.__name__[:-11]] = eval_results
    
#     fold_df = pd.DataFrame(eval_report)
#     # print(fold_df)
#     cv_fold_results.append(eval_report)

# # ================= CROSS-VALIDATION SUMMARY =================
# print("\n" + "="*75)
# print("CROSS-VALIDATION SUMMARY (Mean ± Std Dev)")
# print("="*75)

# # Organize results by model and metric
# models = list(cv_fold_results[0].keys())
# metrics = list(cv_fold_results[0][models[0]].keys())

# cv_summary = {}
# for model_name in models:
#     cv_summary[model_name] = {}
#     for metric in metrics:
#         scores = [fold_results[model_name][metric] for fold_results in cv_fold_results]
#         mean_score = np.mean(scores)
#         std_score = np.std(scores)
#         cv_summary[model_name][metric] = f"{mean_score:.4f} ± {std_score:.4f}"

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# summary_df = pd.DataFrame(cv_summary)
# print(summary_df)

# ================= HYPERPARAMETER EXPERIMENTS =================
print("\n--- Hyperparameter Experiments ---")
import os
from sklearn.metrics import root_mean_squared_error

os.makedirs('figures', exist_ok=True)

# 1. Vary min_support
min_support_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
fixed_params_1 = {'n_neighbors': 25, 'center_ratings': False}

results_min_support = {k: [] for k in ['precision@10', 'recall@10', 'f1@10', 'MSE', 'RMSE']}

for val in min_support_values:
    print(f"Testing min_support = {val}...")
    fold_metrics = {k: [] for k in results_min_support.keys()}
    
    for train_df, test_df in load_premade_splits("data"):
        model = UserKNNBasicRecommender(min_support=val, **fixed_params_1)
        model.fit(train_df)
        
        evaluator = RecommenderEvaluator(model, test_df, relevance_threshold=4, prec_rec_k=10)
        eval_results = evaluator.evaluation_report(
            k=10, 
            err_func=[mean_squared_error, root_mean_squared_error]
        )
        
        fold_metrics['precision@10'].append(eval_results['precision@10'])
        fold_metrics['recall@10'].append(eval_results['recall@10'])
        fold_metrics['f1@10'].append(eval_results['f1@10'])
        fold_metrics['MSE'].append(eval_results['mean_squared_error'])
        fold_metrics['RMSE'].append(eval_results['root_mean_squared_error'])
        
    for k in results_min_support.keys():
        results_min_support[k].append(np.mean(fold_metrics[k]))

plt.figure(figsize=(10, 6))
for metric, values in results_min_support.items():
    plt.plot(min_support_values, values, marker='o', label=metric)
plt.title('Metrics vs min_support (n_neighbors=25, center_ratings=False)')
plt.xlabel('min_support')
plt.ylabel('Score/Error')
plt.legend()
plt.grid(True)
plt.savefig('figures/min_support_experiment.png')
plt.close()

# 2. Vary n_neighbors
n_neighbors_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
fixed_params_2 = {'min_support': 5, 'center_ratings': False}

results_n_neighbors = {k: [] for k in ['precision@10', 'recall@10', 'f1@10', 'MSE', 'RMSE']}

for val in n_neighbors_values:
    print(f"Testing n_neighbors = {val}...")
    fold_metrics = {k: [] for k in results_n_neighbors.keys()}
    
    for train_df, test_df in load_premade_splits("data"):
        model = UserKNNBasicRecommender(n_neighbors=val, **fixed_params_2)
        model.fit(train_df)
        
        evaluator = RecommenderEvaluator(model, test_df, relevance_threshold=4, prec_rec_k=10)
        eval_results = evaluator.evaluation_report(
            k=10, 
            err_func=[mean_squared_error, root_mean_squared_error]
        )
        
        fold_metrics['precision@10'].append(eval_results['precision@10'])
        fold_metrics['recall@10'].append(eval_results['recall@10'])
        fold_metrics['f1@10'].append(eval_results['f1@10'])
        fold_metrics['MSE'].append(eval_results['mean_squared_error'])
        fold_metrics['RMSE'].append(eval_results['root_mean_squared_error'])
        
    for k in results_n_neighbors.keys():
        results_n_neighbors[k].append(np.mean(fold_metrics[k]))

plt.figure(figsize=(10, 6))
for metric, values in results_n_neighbors.items():
    plt.plot(n_neighbors_values, values, marker='o', label=metric)
plt.title('Metrics vs n_neighbors (min_support=5, center_ratings=False)')
plt.xlabel('n_neighbors')
plt.ylabel('Score/Error')
plt.legend()
plt.grid(True)
plt.savefig('figures/n_neighbors_experiment.png')
plt.close()
print("Experiments completed. Plots saved in figures/.")