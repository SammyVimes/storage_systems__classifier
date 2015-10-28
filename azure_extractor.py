import sklearn
import pandas as pd
import random
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/res.csv")
# df.iloc[np.random.permutation(len(df))]
df.reset_index(drop=True)

predicted = df["Scored Labels"].tolist()

res = {}
res["y"] = predicted
res["ID"] = [i for i in range(len(predicted))]
result = pd.DataFrame(res)
result.to_csv("data/sol.csv", columns = ['ID', 'y'], index=False)