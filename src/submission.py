import numpy as np
import pandas as pd

if __name__ == "__main__":
    test_candidate_df = pd.read_csv(
        "../input/ieee-bigdata-prepare-dataset/test.csv",
        usecols=["Trip_id", "Destination"],
    )
    sub_df = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/test.csv",
        usecols=["Trip_id"],
    )
    test_pred = np.load("../input/ieee-bigdata-training-cpu/y_test_pred_fold0.npy")
    test_candidate_df["pred"] = test_pred[:, 1]

    sub_df["candidate"] = test_candidate_df.groupby("Trip_id")["Destination"].apply(
        list
    )
    sub_df["preds"] = test_candidate_df.groupby("Trip_id")["pred"].apply(list)
    sub_df["max_id"] = sub_df["preds"].apply(lambda x: np.argmax(x))

    destinations = []
    for cand, max_id in zip(sub_df["candidate"], sub_df["max_id"]):
        destinations.append(cand[max_id])
    sub_df["Destination"] = destinations
    sub_df[["Trip_id", "Destination"]].to_csv("submission.csv", index=False)
