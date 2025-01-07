import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("../github/ML/Titanic/input/train.csv")
test_df = pd.read_csv('../github/ML/Titanic/input/test.csv')
print(train_df.columns.values)
train_df.info()
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

y = train_df["Survived"]
print(y)
features = ["Pclass", "Sex", "SibSp", "Parch"]
x = pd.get_dummies(train_df[features])
x_test = pd.get_dummies(test_df[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x, y)
prediction = model.predict(x_test)
print(prediction)

output = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": prediction})
output.to_csv("../github/ML/Titanic/submission.csv", index=False)
