# Basic KNN Recommender System

This repository contains a from-scratch implementation of a User-Based K-Nearest Neighbors (KNN) Collaborative Filtering recommender system. It was built with the primary goal of learning and understanding the core mechanics of recommendation engines, including model building, evaluation, and hyperparameter tuning.

## Objectives

This project is a hands-on exploration to understand the main mechanisms of the KNN collaborative filtering:
1. User-Item Matrix Construction: How to transform raw user-item-rating logs into a structured, sparse interaction matrix.
2. User Similarity: How to find similar users using distance metrics (like Cosine Similarity) via `sklearn.neighbors.NearestNeighbors`.
3. Rating Prediction (Centering): How to predict a user's rating for an unseen item by calculating a similarity-weighted average of their neighbors' ratings, while correcting for individual user biases by centering user means.
4. Top-K Recommendations: Filtering valid items (with enough minimum support from neighbors), scoring them, and generating a ranked list of top-K recommendations.
5. Evaluation Metrics: Implementing and interpreting ranking metrics (Precision@k, Recall@k, F1@k) and prediction errors (MSE, RMSE).
6. Hyperparameter Tuning: Observing how changes to $k$ (`n_neighbors`), `min_support` (minimum number of neighbors rating an item), and `center_ratings` affect system performance.

## Project Structure

- `src/models/knn_cf_basic.py`: Contains the implementation of `UserKNNBasicRecommender` class.
- `src/models/item_mean.py`: An item-mean baseline model for comparing performance.
- `src/evaluator.py`: Custom evaluation logic for metrics like precision, recall, and RMSE.
- `src/data_splits.py`: Data loading and cross-validation split utilities.
- `knn_collaborative_filtering.ipynb`: The entry point for running the models, cross-validation, and generating hyperparameter experiment plots.
- `data/`: Directory for the datset (e.g., MovieLens 100K `u.data`).
- `figures/`: Output folder for hyperparameter experiment plots.

## How to Run

1. Ensure you have the required dependencies installed (`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`).
2. Make sure the dataset (e.g. MovieLens `u.data`) is present inside the `data/` directory.
3. Run the main evaluation loop & hyperparameter experiments:
   ```bash
   python knn_cf_test.py
   ```
   Alternatively, you can run the notebook `knn_collaborative_filtering.ipynb`. It contains mostly the same code in knn_cf_test.py, but more interactive
4. Check the console output for Cross-Validation summaries and the `figures/` directory for generated plots showing the impact of `min_support` and `n_neighbors`.

## Key Learnings & Takeaways

- The user-item matrix is vastly sparse. Utilizing sparse matrix representations (like `csr_matrix`) is necessary for memory and processing efficiency.
- Simply taking the mean rating of a few neighbors can lead to unpredictable recommendations. Setting a `min_support` threshold (a minimum number of neighbors who must have rated an item) improves robustness significantly.
- Users have different baseline rating tendencies (some are harsh, some are generous). Centering ratings (subtracting a user's mean rating before calculating similarities and additions) produces more accurate predictions than using raw ratings.
- Tuning hyperparameters often reveals clear trade-offs between precision and recall, as well as finding the "sweet spot" for nearest-neighbor count $k$ to minimize prediction errors like RMSE.
