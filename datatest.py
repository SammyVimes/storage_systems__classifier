# -*- coding: utf-8 -*-
import sklearn
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("C:\\Users\\Семён\\PycharmProjects\\storage_systems__classifier\\data\\train.csv")
# df.iloc[np.random.permutation(len(df))]
df.reset_index(drop=True)

cols = list(df.columns)
cols[8] = "x21"
cols[13] = "x22"
cols[21] = "x8"
cols[22] = "x13"
df = df[cols]

df_train = df.ix[:, 'x0':'x2'].as_matrix()

rows, columns = df_train.shape


def coo_to_sparse_DF(m, sz):
    return pd.SparseDataFrame([pd.SparseSeries(m[i].toarray().ravel()) for i in np.arange(sz)])

def build_col_dict(column, column2):
    labels = set()
    for label in column:
        labels.add(label)
    for label in column2:
        labels.add(label)
    return {y: x for x, y in enumerate(list(labels))}


mapped = []
dicts = []

test_set = pd.read_csv("C:\\Users\\Семён\\PycharmProjects\\storage_systems__classifier\\data\\test.csv")
test_set.reset_index(drop=True)
cols = list(test_set.columns)
cols[9] = "x21"
cols[14] = "x22"
cols[22] = "x8"
cols[23] = "x13"
test_set = test_set[cols]

test_set_to_encode = test_set.ix[:, 'x0':'x2'].as_matrix()

for col in range(columns):
    coldict = build_col_dict(df_train[:, col], test_set_to_encode[:, col])
    newcolumn = [coldict[l] for l in df_train[:, col]]
    mapped.append(coldict)
    df_train[:, col] = newcolumn

df.ix[:, 'x0':'x2'] = df_train


enc = OneHotEncoder(categorical_features=range(2))
matr = df.ix[:, 0:2].as_matrix()

rows, columns = test_set_to_encode.shape
for col in range(columns):
    coldict = mapped[col]
    newcolumn = [coldict[l] for l in test_set_to_encode[:, col]]
    test_set_to_encode[:, col] = newcolumn
test_set.ix[:, 'x0':'x2'] = test_set_to_encode
test_matr = test_set.ix[:, 'x0':'x2'].as_matrix()
ml = len(matr)

matr = np.append(matr, test_matr, axis=1)

newdf = enc.fit_transform(matr)

#TODO: не забыть ограничивать размер!! по ML

print("before to sparse")

df_cat = coo_to_sparse_DF(newdf.tocsr(), ml)
print("before concat")

df = pd.concat([df_cat, df.ix[:, 21:61]], axis=1)


print("before to csv")

df.to_csv("C:\\Users\\Семён\\PycharmProjects\\storage_systems__classifier\\data\\encoded.csv", index=False)