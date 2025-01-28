import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

import data_init
import module_train
import test_output

def trainDfPrint(train_df):
    plt.hist(train_df.loc[train_df.efs==1,"y"],bins=100,label="efs=1, Yes Event")
    plt.hist(train_df.loc[train_df.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
    plt.xlabel("Transformed Target y")
    plt.ylabel("Density")
    plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
    plt.legend()
    plt.show()

train_df, test_df = data_init.dataReadIn()

train_df["y"] = data_init.targetsTransform(train_df, time_col = 'efs_time', event_col = 'efs')

RMV = ["ID","efs","efs_time","y"]
feature_df = [c for c in train_df.columns if not c in RMV]
data_init.initCats(train_df, test_df, feature_df)

train_df, test_df = data_init.labelEncodeCategorical(train_df, test_df, feature_df)
