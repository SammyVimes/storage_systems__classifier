# -*- coding: utf-8 -*-
import sklearn
import pandas as pd
import random
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/train.csv")
df.iloc[np.random.permutation(len(df))]
df.reset_index(drop=True)


mapped = []
dicts = []

test_set = pd.read_csv("data/test.csv")
test_set.reset_index(drop=True)


def encode_onehot(df, cols):
    """
    One-hot encoding применяется к каждой колонке
    каждое значение категориальной фичи
    становится "новой фичей"

    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()

    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df, vec


test_set = test_set.drop("ID", axis=1)
train_true = df['y']
train_set = df.drop("y", axis=1)

# Все фичи, не только категориальные (числовые не будут векторизованы)
cols = ["x" + str(i) for i in range(0, 62)]
encoded, vectorizer = encode_onehot(train_set, cols=cols)

print("train encoded")

train_set = encoded.join(train_true)
train_set.to_csv("data/encoded_train.csv")

print("train saved as CSV")

vec_data = pd.DataFrame(vectorizer.transform(test_set[cols].to_dict(outtype='records')).toarray())
print("test encoded")
vec_data.columns = vectorizer.get_feature_names()
vec_data.index = df.index
test_set = test_set.drop(cols, axis=1)
test_set = test_set.join(vec_data)
print("test joined")

test_set.to_csv("data/encoded_test.csv")

print("done")
