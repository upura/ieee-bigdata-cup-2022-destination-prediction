import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer


def preprocessing_df(input_df):
    output_df = input_df.copy()
    # extract datetime features
    # todo: holiday info
    output_df["Departure_time"] = pd.to_datetime(output_df["Departure_time"])
    output_df["Departure_hour"] = output_df["Departure_time"].dt.hour
    output_df["Departure_dow"] = output_df["Departure_time"].dt.weekday
    return output_df


def create_neg_df(input_df):
    des_neg = []
    output_df = input_df.copy()
    top_two_zone = output_df["Origin"].value_counts().index[:2]
    for ori, des in zip(output_df["Origin"], output_df["Destination"]):
        if des != ori:
            des_neg.append(ori)
        elif des != top_two_zone[0]:
            des_neg.append(top_two_zone[0])
        elif des != top_two_zone[1]:
            des_neg.append(top_two_zone[1])
    output_df["Destination"] = des_neg
    return output_df


def create_train_df(input_df):
    output_df = input_df.copy()
    input_df_neg = create_neg_df(output_df)
    output_df["target"] = 1
    input_df_neg["target"] = 0
    return pd.concat([output_df, input_df_neg]).reset_index(drop=True)


if __name__ == "__main__":
    train_tokyo = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/train/train/Tokyo.csv"
    )
    train_chukyo = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/train/train/Chukyo.csv"
    )
    train_kyushu = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/train/train/Kyushu.csv"
    )
    train_higashisurugawan = pd.read_csv(
        "/kaggle/input/ieee-bigdata-cup-2022-destination-prediction/train/train/Higashisurugawan.csv"
    )
    test = pd.read_csv(
        "/kaggle/input/ieee-bigdata-cup-2022-destination-prediction/test.csv"
    )

    print(
        train_tokyo.shape,
        train_chukyo.shape,
        train_kyushu.shape,
        train_higashisurugawan.shape,
        test.shape,
    )
    # ((790613, 8), (205435, 8), (343752, 8), (34496, 8), (967459, 8))
    print(
        train_tokyo.query("Origin==Destination").shape,
        train_chukyo.query("Origin==Destination").shape,
        train_kyushu.query("Origin==Destination").shape,
        train_higashisurugawan.query("Origin==Destination").shape,
    )
    # ((211518, 8), (52470, 8), (95345, 8), (10638, 8))

    tokyo_feat = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/Zone_features/Zone_features/Tokyo_zone_feature_area.csv"
    )
    chukyo_feat = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/Zone_features/Zone_features/Chukyo_zone_feature_area.csv"
    )
    kyushu_feat = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/Zone_features/Zone_features/Kyushu_zone_feature_area.csv"
    )
    higashisurugawan_feat = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/Zone_features/Zone_features/Higashisurugawan_zone_feature_area.csv"
    )
    test_feat = pd.read_csv(
        "../input/ieee-bigdata-cup-2022-destination-prediction/Zone_features/Zone_features/Kinki_zone_feature_area.csv"
    )
    feat_df = pd.concat(
        [tokyo_feat, chukyo_feat, kyushu_feat, higashisurugawan_feat, test_feat]
    )
    num_cols = ["T000918002", "T000918006", "T000918021", "T000918025", "T000847001"]
    prep = QuantileTransformer(output_distribution="normal")
    prep.fit(feat_df[num_cols])

    train_tokyo = preprocessing_df(train_tokyo)
    train_chukyo = preprocessing_df(train_chukyo)
    train_kyushu = preprocessing_df(train_kyushu)
    train_higashisurugawan = preprocessing_df(train_higashisurugawan)
    test = preprocessing_df(test)

    train_test = pd.concat(
        [train_tokyo, train_chukyo, train_kyushu, train_higashisurugawan, test]
    )
    for c in ["Gender", "Age", "Occupation", "Trip_type"]:
        le = LabelEncoder()
        le.fit(train_test[c].astype(str).values)
        train_tokyo[c] = le.transform(train_tokyo[c].astype(str).values)
        train_chukyo[c] = le.transform(train_chukyo[c].astype(str).values)
        train_kyushu[c] = le.transform(train_kyushu[c].astype(str).values)
        train_higashisurugawan[c] = le.transform(
            train_higashisurugawan[c].astype(str).values
        )
        test[c] = le.transform(test[c].astype(str).values)

    feat_df.loc[:, num_cols] = pd.DataFrame(
        prep.transform(feat_df[num_cols]), columns=num_cols
    )

    train_tokyo = create_train_df(train_tokyo)
    train_chukyo = create_train_df(train_chukyo)
    train_kyushu = create_train_df(train_kyushu)
    train_higashisurugawan = create_train_df(train_higashisurugawan)

    train_tokyo.to_csv("train_tokyo.csv", index=False)
    train_chukyo.to_csv("train_chukyo.csv", index=False)
    train_kyushu.to_csv("train_kyushu.csv", index=False)
    train_higashisurugawan.to_csv("train_higashisurugawan.csv", index=False)
    test.to_csv("test.csv", index=False)
    feat_df.to_csv("zone_features.csv", index=False)
