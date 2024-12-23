import pandas as pd
import numpy as np

note = "CU_conf0.2_4o_mini_newHS"
mistakes_path = "./save/" + note + "_mistakes.npz"
mistakes = np.load(mistakes_path, allow_pickle=True)
print(mistakes["data"])

eval_data = pd.read_csv(
    "/home/yifan40/Multidimensional-MB/data/reddit_data/eval_dataset(Nov5).csv"
)
print(eval_data["hate speech"].value_counts())
