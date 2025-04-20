import pandas as pd
import numpy as np
import os


def stratify_birads_density(base_df_path, annotations_df_path, output_path):
    """
    Stratifies a dataset with respect to both breast_birads and breast_density.
    Adds rows from annotations_df to balance representation while maintaining unique image_ids.

    Args:
        base_df_path (str): Path to the base dataframe CSV that's already somewhat balanced
        annotations_df_path (str): Path to the annotations dataframe with additional findings
        output_path (str): Path where to save the stratified dataframe
    """
    print("Loading datasets...")
    # Load the already balanced dataset
    df_base = pd.read_csv(base_df_path)
    # Load the annotations with additional findings
    df_annotations = pd.read_csv(annotations_df_path)

    print(f"Base dataset: {len(df_base)} rows")
    print(f"Annotations dataset: {len(df_annotations)} rows")

    # Check for duplicate image_ids within the base dataset
    duplicate_base_images = df_base[df_base.duplicated(subset=["image_id"], keep=False)]
    print(f"\nDuplicate image_ids in base dataset: {len(duplicate_base_images)}")

    # Option 1: Keep only one instance of each image_id in the base dataset
    # This is safer but will reduce your dataset size
    # df_base = df_base.drop_duplicates(subset=['image_id'], keep='first')

    # Option 2: Create a composite unique ID using both study_id and image_id
    # This preserves all data but assumes each study_id + image_id combination should be unique
    if "study_id" in df_base.columns:
        print("Creating composite IDs from study_id and image_id")
        df_base["composite_id"] = (
            df_base["study_id"] + "_" + df_base["image_id"].astype(str)
        )
        df_annotations["composite_id"] = (
            df_annotations["study_id"] + "_" + df_annotations["image_id"].astype(str)
        )
        id_column = "composite_id"
    else:
        print("No study_id column found, using image_id alone")
        id_column = "image_id"

    # Analyze current distribution of breast_birads and breast_density in base dataset
    birads_counts = df_base["breast_birads"].value_counts()
    density_counts = df_base["breast_density"].value_counts()
    combined_counts = (
        df_base.groupby(["breast_birads", "breast_density"])
        .size()
        .reset_index(name="count")
    )

    print("\nCurrent BIRADS distribution:")
    print(birads_counts)

    print("\nCurrent breast density distribution:")
    print(density_counts)

    # Show the most underrepresented combinations
    print("\nMost underrepresented combinations (BIRADS, Density):")
    print(combined_counts.sort_values("count").head(10))

    # Find the target counts (max counts for each category)
    max_birads_count = birads_counts.max()
    max_density_count = density_counts.max()

    # Get existing IDs to avoid duplicates
    existing_ids = set(df_base[id_column].unique())

    # Filter annotations to exclude existing IDs
    unique_annotations = df_annotations[
        ~df_annotations[id_column].isin(existing_ids)
    ].copy()
    print(
        f"\nUnique annotations available for addition: {len(unique_annotations)} rows"
    )

    # IMPROVEMENT: Check if we have enough data for each category
    birads_available = unique_annotations.groupby("breast_birads").size()
    density_available = unique_annotations.groupby("breast_density").size()

    print("\nAvailable samples for each BIRADS:")
    print(birads_available)

    print("\nAvailable samples for each density:")
    print(density_available)

    # Create a dictionary to track how many rows we need to add for each category
    birads_to_add = {
        birads: max_birads_count - count for birads, count in birads_counts.items()
    }
    density_to_add = {
        density: max_density_count - count for density, count in density_counts.items()
    }

    print("\nBIRADS rows to add:")
    print(birads_to_add)
    print("\nDensity rows to add:")
    print(density_to_add)

    # IMPROVEMENT: Check feasibility of balancing
    for birads, needed in birads_to_add.items():
        if needed > 0:
            available = birads_available.get(birads, 0)
            if available < needed:
                print(
                    f"WARNING: Not enough samples for BIRADS {birads}. Need {needed}, have {available}."
                )

    for density, needed in density_to_add.items():
        if needed > 0:
            available = density_available.get(density, 0)
            if available < needed:
                print(
                    f"WARNING: Not enough samples for density {density}. Need {needed}, have {available}."
                )

    # Initialize an empty DataFrame to hold rows to be added
    rows_to_add = pd.DataFrame()

    # IMPROVEMENT: Prioritize underrepresented combinations instead of separate balancing
    combined_to_add = {}
    for _, row in combined_counts.iterrows():
        birads = row["breast_birads"]
        density = row["breast_density"]
        count = row["count"]
        target_count = min(
            max_birads_count, max_density_count
        )  # Set a reasonable target
        if count < target_count:
            combined_to_add[(birads, density)] = target_count - count

    # Sort combinations by how underrepresented they are (largest deficit first)
    sorted_combinations = sorted(
        combined_to_add.items(), key=lambda x: x[1], reverse=True
    )

    print("\nTarget combinations to balance (most underrepresented first):")
    for (birads, density), needed in sorted_combinations[:10]:  # Show top 10
        print(f"BIRADS: {birads}, Density: {density}, Need: {needed}")

    # Add samples for underrepresented combinations first
    for (birads, density), needed_count in sorted_combinations:
        if needed_count <= 0:
            continue

        # Find eligible rows in unique_annotations for this combination
        eligible_rows = unique_annotations[
            (unique_annotations["breast_birads"] == birads)
            & (unique_annotations["breast_density"] == density)
        ]

        if len(eligible_rows) == 0:
            print(f"No eligible rows found for BIRADS {birads}, Density {density}")
            continue

        # Sample min(needed_count, available count) rows
        sample_count = min(needed_count, len(eligible_rows))
        sampled_rows = eligible_rows.sample(n=sample_count, random_state=42)

        # Update unique_annotations to avoid reusing the same rows
        used_ids = set(sampled_rows[id_column])
        unique_annotations = unique_annotations[
            ~unique_annotations[id_column].isin(used_ids)
        ]

        # Append sampled rows to rows_to_add
        rows_to_add = pd.concat([rows_to_add, sampled_rows], ignore_index=True)

        print(f"Added {len(sampled_rows)} rows for BIRADS {birads}, Density {density}")

    # If we still need more samples for individual categories, do additional sampling
    # First, sample remaining BIRADS deficits
    remaining_birads_to_add = {
        birads: need for birads, need in birads_to_add.items() if need > 0
    }

    for birads, needed_count in remaining_birads_to_add.items():
        eligible_rows = unique_annotations[
            unique_annotations["breast_birads"] == birads
        ]

        if len(eligible_rows) == 0:
            continue

        sample_count = min(needed_count, len(eligible_rows))
        sampled_rows = eligible_rows.sample(n=sample_count, random_state=42)

        used_ids = set(sampled_rows[id_column])
        unique_annotations = unique_annotations[
            ~unique_annotations[id_column].isin(used_ids)
        ]

        rows_to_add = pd.concat([rows_to_add, sampled_rows], ignore_index=True)
        print(f"Added {len(sampled_rows)} more rows for BIRADS {birads}")

    # Then, sample remaining density deficits
    remaining_density_to_add = {
        density: need for density, need in density_to_add.items() if need > 0
    }

    for density, needed_count in remaining_density_to_add.items():
        eligible_rows = unique_annotations[
            unique_annotations["breast_density"] == density
        ]

        if len(eligible_rows) == 0:
            continue

        sample_count = min(needed_count, len(eligible_rows))
        sampled_rows = eligible_rows.sample(n=sample_count, random_state=42)

        rows_to_add = pd.concat([rows_to_add, sampled_rows], ignore_index=True)
        print(f"Added {len(sampled_rows)} more rows for Density {density}")

    # Add marker columns to identify added rows
    rows_to_add["added_for_stratification"] = True

    # Add the marker column to the base dataframe (all false)
    df_base["added_for_stratification"] = False

    # Combine base dataframe with added rows
    df_stratified = pd.concat([df_base, rows_to_add], ignore_index=True)

    # Final stats
    print("\nFinal dataset statistics:")
    print(f"Total rows: {len(df_stratified)}")
    print(f"Original rows: {len(df_base)}")
    print(f"Rows added for stratification: {len(rows_to_add)}")

    print("\nFinal BIRADS distribution:")
    print(df_stratified["breast_birads"].value_counts())

    print("\nFinal breast density distribution:")
    print(df_stratified["breast_density"].value_counts())

    # Check for duplicate image_ids
    duplicate_image_ids = df_stratified[
        df_stratified.duplicated(subset=["image_id"], keep=False)
    ]
    if len(duplicate_image_ids) > 0:
        print(
            f"\nWARNING: Found {len(duplicate_image_ids)} rows with duplicate image_ids!"
        )

        # IMPROVEMENT: Add analysis of duplicates
        duplicate_counts = duplicate_image_ids["image_id"].value_counts()
        print(f"Number of unique images with duplicates: {len(duplicate_counts)}")
        print(f"Max duplicates for a single image: {duplicate_counts.max()}")

        # Option to handle duplicates if needed
        print("\nOptions to handle duplicates:")
        print("1. Keep them (multiple findings per image)")
        print("2. Create a file with study_id/image_id.png paths for proper loading")
    else:
        print("\nNo duplicate image_ids found, all good!")

    # IMPROVEMENT: Create a file paths mapping
    if len(duplicate_image_ids) > 0 and "study_id" in df_stratified.columns:
        print("\nGenerating image path mapping file for proper loading...")

        # Create a column with the proper file path
        df_stratified["file_path"] = df_stratified.apply(
            lambda row: os.path.join(
                str(row["study_id"]), str(row["image_id"]) + ".png"
            ),
            axis=1,
        )

        # Save the file paths to a separate file
        paths_file = output_path.replace(".csv", "_file_paths.csv")
        df_stratified[["image_id", "study_id", "file_path"]].to_csv(
            paths_file, index=False
        )
        print(f"Image paths mapping saved to {paths_file}")

    # Save the stratified dataset
    df_stratified.to_csv(output_path, index=False)
    print(f"\nStratified dataset saved to {output_path}")

    return df_stratified


if __name__ == "__main__":
    # File paths
    base_df_path = "../metadata/base/stratified_local_balanced_v2.csv"
    annotations_df_path = "../metadata/base/finding_annotations.csv"
    output_path = "../metadata/classification/stratified_birads_density_balanced.csv"

    # Run stratification
    stratified_df = stratify_birads_density(
        base_df_path, annotations_df_path, output_path
    )
