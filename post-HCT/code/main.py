import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

import data_init
import module_train
import test_output

train_df = data_init.dataInit()

train_df["y"] = data_init.targetsTransform(train_df, time_col = 'efs_time', event_col = 'efs')
plt.hist(train_df.loc[train_df.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train_df.loc[train_df.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()