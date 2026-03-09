import numpy as np
scores = np.array([1.0, 5.0, np.nan, 2.0])
top_indices = np.argsort(scores)[::-1]
print(f"Scores: {scores}")
print(f"Top indices: {top_indices}")
print(f"Scores at top indices: {scores[top_indices]}")
