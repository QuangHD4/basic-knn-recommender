import pandas as pd

def load_premade_splits(data_dir):
    """
    Load pre-made 5-fold splits from MovieLens 100K dataset.
    """
    for i in range(1, 6):
        train_path = f"{data_dir}/u{i}.base"
        test_path = f"{data_dir}/u{i}.test"

        train_df = pd.read_csv(train_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
        test_df = pd.read_csv(test_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

        yield train_df, test_df


def temporal_user_split(df, test_ratio=0.2):

    df = df.sort_values("timestamp")

    train_list = []
    test_list = []

    for user, group in df.groupby("user_id"):

        n_test = max(1, int(len(group) * test_ratio))

        test = group.tail(n_test)
        train = group.iloc[:-n_test]

        if len(train) == 0:
            continue

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df