import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


bias_dim = [
    "hate speech",
    "linguistic bias",
    "text-level context bias",
    "political bias",
    "gender bias",
    "racial bias",
]
domains = ["healthcare", "job_education", "sports", "entertainment", "politics"]
annotators = ["lydia", "alex"]


if __name__ == "__main__":
    data = {"lydia": [], "alex": []}
    # concatenate data for each annotator, regardless of domain
    for annotator in annotators:
        for domain in domains:
            df = None
            try:
                df = pd.read_excel(f"./{annotator}_{domain}.xlsx")
                data[annotator].append(df)
            except:
                print(f"File {annotator}_{domain} not found.")
                continue

    annotator1_df = pd.concat(data["lydia"])
    annotator2_df = pd.concat(data["alex"])
    annotator1_df = annotator1_df.dropna(subset=["hate speech"])
    annotator2_df = annotator2_df.dropna(subset=["hate speech"])
    print(len(annotator1_df), len(annotator2_df))

    # Perform inner join based on "id"
    merged_df = pd.merge(
        annotator1_df, annotator2_df, on="Unnamed: 0", suffixes=("_lydia", "_alex")
    )
    print(len(merged_df))

    """
    annotator1_df = annotator1_df.sort_values(by="id")
    annotator2_df = annotator2_df.sort_values(by="id")
    common_ids = set(annotator1_df["Unnamed: 0"]).intersection(
        set(annotator2_df["Unnamed: 0"])
    )
    print(len(common_ids))
    # annotator1_df = annotator1_df.isin(common_ids).sort_values(by="id")
    # annotator2_df = annotator2_df.isin(common_ids).sort_values(by="id")

    print(len(annotator1_df), len(annotator2_df))
    # check cohen's kappa for common ids
    for bias in bias_dim:
        df1 = np.asarray(annotator1_df[bias])
        df2 = np.asarray(annotator2_df[bias])
        print(len(df1), len(df2))
        # calculate cohen's kappa
        kappa = cohen_kappa_score(df1, df2)
        print(f"Cohen's kappa for {bias} is: {kappa}")
    """
