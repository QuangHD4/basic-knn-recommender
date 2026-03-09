import numpy as np

def low_coverage_check(train_df, test_df, recommender):
    # low item overlap
    train_items = set(train_df.item_id)
    test_items = set(test_df.item_id)
    item_overlap = len(train_items & test_items) / len(test_items)
    print("Item overlap:", item_overlap)

    # low user overlap
    train_users = set(train_df.user_id)
    test_users = set(test_df.user_id)
    user_overlap = len(train_users & test_users) / len(test_users)
    print("User overlap:", user_overlap)

    # no neighbors with rating for item
    no_neighbor_rating = 0
    total_checked = 0

    for row in test_df.itertuples(index=False):

        user_id = row.user_id
        item_id = row.item_id

        if user_id not in recommender.user_index:
            continue

        if item_id not in recommender.item_index:
            continue

        total_checked += 1

        u = recommender.user_index[user_id]
        i = recommender.item_index[item_id]

        # get neighbor indices directly (faster than calling get_neighbors)
        neighbor_idxs = recommender.neighbor_indices[u]

        # ratings of neighbors for this item
        neighbor_ratings = recommender.user_item[neighbor_idxs, i]

        # check if any neighbor rated the item
        if np.all(np.isnan(neighbor_ratings)):
            no_neighbor_rating += 1


    print(
        "No-neighbor cases:",
        no_neighbor_rating / total_checked if total_checked else 0
    )