import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


def construct_eval_dataset(annotators, domains):
    data = {annotator: [] for annotator in annotators}
    for annotator in annotators:
        for domain in domains:
            df = None
            try:
                df = pd.read_excel(f"./{annotator}_{domain}.xlsx")
                df["domain"] = domain  # Add a column indicating domain
                data[annotator].append(df)
            except FileNotFoundError:
                print(f"File {annotator}_{domain} not found.")
                continue

    combined_data = []
    for annotator in annotators:
        data[annotator] = pd.concat(data[annotator])
        combined_data.append(data[annotator])

    combined_df = pd.concat(combined_data)
    combined_df = combined_df.dropna(subset=["hate speech"])
    duplicates = combined_df[combined_df["id"].duplicated()]
    print(f"Number of rows before removing duplicates: {len(combined_df)}")
    if not duplicates.empty:
        print(f"Duplicate rows based on 'id': {duplicates['id'].values}")
        combined_df = combined_df.drop_duplicates(subset=["id"], keep="first")
    print(f"Number of rows after removing duplicates: {len(combined_df)}")
    assert combined_df["id"].duplicated().sum() == 0
    combined_df = combined_df.drop(columns=["Unnamed: 0"])
    return combined_df


def calculate_kappa_scores(file_paths_1, file_paths_2):
    bias_dim = [
        "hate speech",
        "linguistic bias",
        "text-level context bias",
        "gender bias",
        "political bias",
        "racial bias",
    ]
    # Determine file type and read the files accordingly
    data_1 = []
    data_2 = []
    for i in range(len(file_paths_1)):
        file_path_1 = file_paths_1[i]
        file_path_2 = file_paths_2[i]
        if file_path_1.endswith(".csv") and file_path_2.endswith(".csv"):
            df1 = pd.read_csv(file_path_1)
            df2 = pd.read_csv(file_path_2)
        elif file_path_1.endswith(".xlsx") and file_path_2.endswith(".xlsx"):
            df1 = pd.read_excel(file_path_1)
            df2 = pd.read_excel(file_path_2)
        else:
            raise ValueError("Files must be either both CSV or both Excel files")
        data_1.append(df1)
        data_2.append(df2)

    df1 = pd.concat(data_1)
    df2 = pd.concat(data_2)

    df1 = df1.sort_values(by="id").reset_index(drop=True)
    df2 = df2.sort_values(by="id").reset_index(drop=True)

    # Calculate Cohen's kappa for each column
    kappa_scores = {}
    for column in bias_dim:
        kappa = cohen_kappa_score(df1[column], df2[column])
        kappa_scores[column] = kappa

    print(len(df1["hate speech"]))
    print(len(df2["hate speech"]))
    # Print the kappa scores
    print("=== Cohen's kappa scores ===")
    for column, kappa in kappa_scores.items():
        if np.isnan(kappa) and (list(df1[column]) == list(df2[column])):
            kappa = 1.0
        print(f"Cohen's kappa for {column}: {kappa}")


if __name__ == "__main__":
    calculate_kappa_scores(
        [
            "/home/yifan/Desktop/tidy/PapersNProjects/Code/MMediaBias/Multidimensional-MB/data/reddit_data/alex_v1.csv",
            "/home/yifan/Desktop/tidy/PapersNProjects/Code/MMediaBias/Multidimensional-MB/data/reddit_data/alex_v2.xlsx",
        ],
        [
            "/home/yifan/Desktop/tidy/PapersNProjects/Code/MMediaBias/Multidimensional-MB/data/reddit_data/lydia_v1.csv",
            "/home/yifan/Desktop/tidy/PapersNProjects/Code/MMediaBias/Multidimensional-MB/data/reddit_data/lydia_v2.xlsx",
        ],
    )

    # construct dataset for evaluation
    annotators = ["lydia", "alex"]
    domains = ["healthcare", "job_education", "sports", "entertainment", "politics"]
    eval_dataset = construct_eval_dataset(annotators, domains)
    eval_dataset.to_csv("eval_dataset(Nov5).csv")
