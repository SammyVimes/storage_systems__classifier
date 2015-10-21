import sklearn
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/train.csv")
#df.iloc[np.random.permutation(len(df))]
df.reset_index(drop=True)

df_train = df.ix[:, 'x0':'x23'].as_matrix()

#
#На 8 месте - float
#На 13 месте -- то ли int то ли категориальные
#Возможно, NaN надо обрабатывать отдельно -- каждый раз как новая категория, а может и нет.
#

rows, columns = df_train.shape


def build_col_dict(column):
    labels = set()    
    for label in column:
        labels.add(label)
    return {y:x for x,y in enumerate(list(labels))}

mapped = []
dicts = []

for col in range(columns):
    coldict = build_col_dict(df_train[:, col])
    newcolumn = [coldict[l] for l in df_train[:, col]]
    # newcolumn = map(lambda l : coldict[l], df_train[:, col])
    mapped.append(coldict)
    df_train[:, col] = newcolumn


df.ix[:, 'x0':'x23'] = df_train


enc = OneHotEncoder(categorical_features=range(23))
matr = df.ix[:, 0:23].as_matrix()
newdf = enc.fit_transform(matr)


mmm = {}
for i in range(newdf.shape[1]):
    mmm[str(i)] = newdf[:, i]

df_cat = pd.DataFrame(mmm, orient='column')
