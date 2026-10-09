import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(input_file):
    print("--- PRACTICAL 6: Dataset Splitting (Train/Test Split) ---\n")

    print(f"Loading dataset from '{input_file}'...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Total Rows Loaded: {len(df):,}")

    feature_columns = [
        "ip_request_count", "time_diff_seconds", "status_code_404_count", "is_bot_agent", "url_entropy"
    ]
    X = df[feature_columns]
    y = df["label"]

    print("\n--- Selected Feature Columns (X) ---")
    print(list(X.columns))

    print("\nSplitting dataset into 80% Training set and 20% Testing set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    try:
        from imblearn.over_sampling import SMOTE

        print("\n[Balancing] Applying SMOTE (oversampling) to the training set...")
        X_train_balanced, y_train_balanced = SMOTE(random_state=42).fit_resample(X_train, y_train)
    except ImportError:
        print("\n'imblearn' not installed. Applying random undersampling instead...")
        train_df = X_train.copy()
        train_df["label"] = y_train
        min_class_size = train_df["label"].value_counts().min()
        balanced_frames = [
            train_df[train_df["label"] == label].sample(n=min_class_size, random_state=42)
            for label in train_df["label"].unique()
        ]
        balanced_train_df = pd.concat(balanced_frames).sample(frac=1, random_state=42).reset_index(drop=True)
        X_train_balanced = balanced_train_df.drop(columns=["label"])
        y_train_balanced = balanced_train_df["label"]

    print("\n--- Balanced Training Set Label Distribution ---")
    for label, count in y_train_balanced.value_counts().items():
        percentage = count / len(y_train_balanced) * 100
        print(f"  * {str(label).upper():<15}: {count:,} ({percentage:.2f}%)")

    train_output = X_train_balanced.copy()
    train_output["label"] = y_train_balanced
    test_output = X_test.copy()
    test_output["label"] = y_test
    train_output.to_csv("train_dataset.csv", index=False)
    test_output.to_csv("test_dataset.csv", index=False)
    print("\nSaved balanced 'train_dataset.csv' and 'test_dataset.csv' successfully!")


if __name__ == "__main__":
    split_dataset("engineered_features.csv")
