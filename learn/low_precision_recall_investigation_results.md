# Investigation: Low Precision and Recall in KNN CF

## 1. Problem Identification
The current implementation of [UserKNNBasicRecommender](file:///d:/Quang/2026_Spring_AIL303m/Capstone1/src/models/knn_cf_basic.py#7-191) shows extremely low Precision@10 (approx. 0.004) and Recall@10 (approx. 0.004), despite having a reasonable MAE and RMSE.

## 2. Root Cause Analysis
Based on the investigation, three main issues have been identified in the [recommend_items](file:///d:/Quang/2026_Spring_AIL303m/Capstone1/src/models/dummies.py#34-46) method:

### A. Bias Towards Niche Items (Main Reason)
The model ranks items based on the weighted average of neighbors' deviations from their means. 
- **Observation**: Items rated by very few neighbors (often just **one**) can receive extremely high scores if that neighbor gave a high rating.
- **Example**: In a test run, many top-recommended items had only **1 neighbor rating** and predicted scores as high as **5.77** (on a 1-5 scale).
- **Impact**: These obscure items are statistically unlikely to be found in the test set, leading to poor Precision and Recall.

### B. Lack of Diversity/Popularity Filtering
Basic KNN Collaborative Filtering does not account for item popularity. While Collaborative Filtering is meant to find personalized items, without a minimum support (minimum number of ratings), it becomes highly susceptible to noise in sparse regions of the user-item matrix.

### C. Implementation Issues
1. **NaN Sorting Bug**: The use of `np.argsort(scores)[::-1]` causes `NaN` scores (items with no neighbor ratings) to be moved to the **front** of the sorting list. Although they are skipped in the loop, this is inefficient and reflects a lack of robustness.
2. **Division Warning**: The code performs division by zero when an item hasn't been rated by any neighbor, causing `RuntimeWarning: invalid value encountered in divide`.

## 3. Recommended Improvements

### Short-term Fixes (Robustness)
- [ ] **Handle NaNs properly**: Filter out or set `NaN` scores to a very low value before sorting.
- [ ] **Fix Division Warning**: Add a small epsilon or check for zero denominators before division.
- [ ] **Rating Clipping**: Clip predicted ratings to the valid range (e.g., [1, 5]).

### Algorithmic Improvements (Performance)
- [ ] **Minimum Neighbors (K-min)**: Only recommend items that have been rated by at least $M$ neighbors (e.g., $M=5$ or $10$).
- [ ] **Shrinkage/Regularization**: Add a constant to the denominator to penalize items with few neighbor ratings (e.g., $denom + \lambda$).
- [ ] **Center Ratings by Default**: Use `center_ratings=True` during similarity computation to implement Pearson Correlation, which is more robust than raw Cosine similarity for recommendation.

## 4. Evidence
Debug output for User ID 1:
```
Item 1368, Score 5.7735, Num neighbor ratings: 1
Item 1643, Score 5.7735, Num neighbor ratings: 1
Item 1449, Score 5.6563, Num neighbor ratings: 1
...
Item 114, Score 4.9350, Num neighbor ratings: 9  <-- A more "supported" recommendation
```
Items with 1 rating predominate the top of the list but are unlikely to be relevant to the general population in the test set.
