import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

CATS = []

def dataReadIn():
    train_df = pd.read_csv("../ML/post-HCT/input/train.csv")
    test_df = pd.read_csv("../ML/post-HCT/input/test.csv")
    return train_df, test_df

from lifelines import KaplanMeierFitter

def targetsTransform(targets, time_col = "efs_time", event_col = "efs"):
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations = targets['efs_time'],          # 时间
        event_observed = targets['efs']     # 事件（或删失）指示
    )
    return kmf.survival_function_at_times(targets[time_col]).values

def labelEncodeCategorical(train_df, test_df, feature_df):
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for c in feature_df:
        if c in CATS :
            print(f"{c}, ",end="")
            combined[c], _ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")
        else:
            if combined[c].dtype == "float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c] = combined[c].astype("int32")
    train = combined.iloc[:len(train_df)].copy()
    test = combined.iloc[len(train_df):].reset_index(drop = True).copy()
    return train, test

def initCats(train_df, test_df, feature_df):
    for c in feature_df:
        if train_df[c].dtype == "object":
            CATS.append(c)
            train_df[c] = train_df[c].fillna("NAN")
            test_df[c] = test_df[c].fillna("NAN")