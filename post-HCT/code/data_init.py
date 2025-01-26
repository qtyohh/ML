import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

def dataInit():
    train_df = pd.read_csv("../ML/post-HCT/input/train.csv")
    print(train_df.columns.values)
    return train_df

from lifelines import KaplanMeierFitter

def targetsTransform(targets, time_col = "efs_time", event_col = "efs"):
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations = targets['efs_time'],          # 时间
        event_observed = targets['efs']     # 事件（或删失）指示
    )
    return kmf.survival_function_at_times(targets[time_col]).values