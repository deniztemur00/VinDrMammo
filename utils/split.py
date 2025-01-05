import json
import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict, Counter
from IPython.display import display
import json
from utils.stratification import IterativeStratification


SEED = 1999
GLOBAL_PATH = "metadata\\breast-level_annotations.csv"
LOCAL_PATH = "metadata\\finding_annotations.csv"
birads_LESIONS = {
    "Mass",
    "Suspicious Calcification",
    "Architectural Distortion",
    "Focal Asymmetry",
    "Global Asymmetry",
    "Asymmetry",
}
NO_BIRADS = {
    "Suspicious Lymph Node",
    "Skin Thickening",
    "Skin Retraction",
    "Nipple Retraction",
    "No Finding",
}
BIRADS345 = ["BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]
ALL_LESIONS = [
    "Suspicious Lymph Node",
    "Mass",
    "Suspicious Calcification",
    "Asymmetry",
    "Focal Asymmetry",
    "Global Asymmetry",
    "Architectural Distortion",
    "Skin Thickening",
    "Skin Retraction",
    "Nipple Retraction",
    "No Finding",
]


def show_df(df):
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
    ):  # more options can be specified also
        display(df)


def count_birads_densities(df):
    """
    count birads density at breast level
    """
    counter = defaultdict(lambda: 0)
    den_counter = defaultdict(lambda: 0)
    for (study_id, side), rows in df.groupby(["study_id", "laterality"]):
        birads = rows.breast_birads.values[0]
        counter[birads] += 1
        density = rows.breast_density.values[0]
        den_counter[density] += 1

    total = sum(counter.values())
    total2 = sum(den_counter.values())
    assert total == total2
    percent = {k: f"{100.*v/total:.2f}" for k, v in counter.items()}
    counter["Total"] = total
    stats = pd.DataFrame.from_records({"No. breast": counter, "percent": percent})
    stats.index.name = "BI-RADS"
    stats = stats.sort_index()

    den_percent = {k: f"{100.*v/total:.2f}" for k, v in den_counter.items()}
    den_counter["Total"] = total
    den_stats = pd.DataFrame({"No. breast": den_counter, "percent": den_percent})
    den_stats.index.name = "DENSITY"
    den_stats = den_stats.sort_index()
    return stats, den_stats


def count_box_birads(df):
    """ """
    counter = defaultdict(lambda: defaultdict(lambda: 0))
    df.finding_birads = df.finding_birads.fillna("")
    all_birads = sorted(df.finding_birads.unique().tolist())
    for _, row in df.iterrows():
        for clas in row.finding_categories:
            counter[clas]["Total"] += 1
            counter[clas][row.finding_birads] += 1
    for k, v in counter.items():
        v["Lesion"] = k
    df = pd.DataFrame.from_records(
        list(counter.values()), columns=["Lesion", "Total"] + all_birads
    )
    lesion = df["Lesion"].values
    df = df.set_index("Lesion")
    df = df.reindex(ALL_LESIONS)

    df = df.fillna(0)
    df.loc["All lesions"] = df.sum()
    df = df.astype("int32")
    return df


def count_box_label(df):
    box_label = list(chain(*df.box_label.tolist()))
    return Counter(box_label)


def df_counts(df):
    print("no. studies", len(df.study_id.unique()))
    print("no. images", len(df.image_id.unique()))


def process_data():
    local_df = pd.read_csv(LOCAL_PATH)
    local_df["finding_categories"] = local_df["finding_categories"].apply(
        lambda x: json.loads(x.replace("'", '"'))
    )
    global_df = pd.read_csv(GLOBAL_PATH)

    split_col = [f"BI-RADS {i}" for i in range(1, 6)]
    split_col = split_col + [f"DENSITY {x}" for x in "ABCD"]
    split_col.extend(list(NO_BIRADS))
    split_col = split_col + [
        f"{box_name}_{box_birads}"
        for box_name in birads_LESIONS
        for box_birads in BIRADS345
    ]
    study_ids = sorted(global_df.study_id.unique().tolist())
    labels_ar = np.zeros((len(study_ids), len(split_col)), dtype=np.int32)
    for (study_id, lat), rows in global_df.groupby(["study_id", "laterality"]):
        birads = rows.breast_birads.values[0]
        density = rows.breast_density.values[0]
        labels_ar[study_ids.index(study_id), split_col.index(birads)] += 1
        labels_ar[study_ids.index(study_id), split_col.index(density)] += 1
    for _, x in local_df.iterrows():
        birads = x["finding_birads"]
        for label in x["finding_categories"]:
            if label in birads_LESIONS:
                labels_ar[
                    study_ids.index(x["study_id"]),
                    split_col.index(f"{label}_{birads}"),
                ] += 1
            else:
                labels_ar[
                    study_ids.index(x["study_id"]),
                    split_col.index(label),
                ] += 1
    total = labels_ar.sum(axis=0)

    SPLITS = np.array([0.8, 0.2])
    stratifier = IterativeStratification(SEED)
    fold_ids = stratifier.stratify(labels_ar, SPLITS)

    global_df["fold"] = ""
    local_df["fold"] = ""
    fold_name = ["training", "test"]
    for k in range(2):
        fold_idx = np.where(fold_ids == k)[0]
        #     print(fold_idx)
        study_uids = [study_ids[i] for i in fold_idx]
        #     print(study_uids[:5])
        global_df.loc[global_df.study_id.isin(study_uids), "fold"] = fold_name[k]
        local_df.loc[local_df.study_id.isin(study_uids), "fold"] = fold_name[k]

    return global_df, local_df


def visualize_df(global_df):
    print("Whole dataset:")

    bi, den = count_birads_densities(global_df)
    show_df(bi)
    show_df(den)

    print("Training split:")
    bi, den = count_birads_densities(global_df[global_df.fold == "training"])
    show_df(bi)
    show_df(den)

    print("Test split:")
    bi, den = count_birads_densities(global_df[global_df.fold == "test"])
    show_df(bi)
    show_df(den)
