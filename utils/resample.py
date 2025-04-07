import pandas as pd
import numpy as np
from sklearn.utils import resample


def resample_minority_classes(
    df,
    target_column="mapped_category",
    birads_column="breast_birads",
    fold_column="fold",
):
    """
    Resamples minority classes in a DataFrame, considering the 'fold' column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the column containing the target classes (e.g., 'mapped_category').
        birads_column (str): The name of the column containing BIRADS information.
        fold_column (str): The name of the column indicating the fold (e.g., 'training', 'test').

    Returns:
        pd.DataFrame: A new DataFrame with resampled minority classes in the training fold,
                      and an 'is_resampled' column.
    """

    # 1. Separate training data
    train_df = df[df[fold_column] == "training"].copy()  # Work on a copy
    original_train_size = len(train_df)

    # 2. Identify minority classes in the training data
    class_counts = (
        train_df.groupby([target_column, birads_column])
        .size()
        .reset_index(name="counts")
    )
    max_count = class_counts["counts"].max()

    # 3. Create a list to store resampled DataFrames
    resampled_dfs = []

    # 4. Iterate through each unique class and BIRADS combination
    for _, row in class_counts.iterrows():
        class_name = row[target_column]
        birads_value = row[birads_column]
        count = row["counts"]

        # 5. Resample if the count is less than the maximum count
        if count < max_count:
            # Filter the training DataFrame to get the samples belonging to this class and BIRADS
            class_df = train_df[
                (train_df[target_column] == class_name)
                & (train_df[birads_column] == birads_value)
            ]

            # Calculate how many samples to add
            n_samples_to_add = max_count - count

            # Resample with replacement to create new samples
            resampled_class_df = resample(
                class_df, replace=True, n_samples=n_samples_to_add, random_state=42
            )

            # Add the 'is_resampled' column
            resampled_class_df["is_resampled"] = True

            # Append the resampled DataFrame to the list
            resampled_dfs.append(resampled_class_df)

    # 6. Concatenate the resampled DataFrames with the original training DataFrame
    if resampled_dfs:
        resampled_train_df = pd.concat(
            [train_df] + resampled_dfs, ignore_index=True
        )
    else:
        resampled_train_df = train_df.copy()

    # 7. Add 'is_resampled' column to original training data and test data
    resampled_train_df["is_resampled"] = resampled_train_df.get(
        "is_resampled", False
    )
    test_df = df[df[fold_column] == "test"].copy()
    test_df["is_resampled"] = False

    # 8. Concatenate training and test DataFrames
    final_df = pd.concat([resampled_train_df, test_df], ignore_index=True)

    return final_df


def main():

    last_df = pd.read_csv("../metadata/final_aggregated_findings_cropped_top3.csv")
    final_df = resample_minority_classes(last_df.copy())


    # Now use 'final_df' in your data loading process
    print(
        final_df.groupby(["mapped_category","fold"])
        .size()
        .reset_index(name="counts")
        .sort_values(by="counts", ascending=False)
    )

    print(
        final_df.groupby(["fold",]).size().reset_index(name="counts")
    )

    print(final_df.describe())
    final_df.to_csv(
        "../metadata/resampled_aggregated_cropped_top3.csv", index=False
    )


if __name__ == "__main__":
    main()
