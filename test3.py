import sklearn
import pandas as pd
import random
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/train.csv")
# df.iloc[np.random.permutation(len(df))]
df.reset_index(drop=True)

cols = list(df.columns)
cols[8] = "x21"
cols[13] = "x22"
cols[21] = "x8"
cols[22] = "x13"
df = df[cols]

df_train = df.ix[:, 'x0':'x20'].as_matrix()

rows, columns = df_train.shape


def build_col_dict(column, column2):
    labels = set()
    for label in column:
        labels.add(label)
    for label in column2:
        labels.add(label)
    return {y: x for x, y in enumerate(list(labels))}


mapped = []
dicts = []

test_set = pd.read_csv("data/test.csv")
test_set.reset_index(drop=True)
cols = list(test_set.columns)
cols[9] = "x21"
cols[14] = "x22"
cols[22] = "x8"
cols[23] = "x13"
test_set = test_set[cols]

test_set_to_encode = test_set.ix[:, 'x0':'x20'].as_matrix()

for col in range(columns):
    coldict = build_col_dict(df_train[:, col], test_set_to_encode[:, col])
    newcolumn = [coldict[l] for l in df_train[:, col]]
    mapped.append(coldict)
    df_train[:, col] = newcolumn

df.ix[:, 'x0':'x20'] = df_train


enc = OneHotEncoder(categorical_features=range(20))
matr = df.ix[:, 0:20].as_matrix()

rows, columns = test_set_to_encode.shape
for col in range(columns):
    coldict = mapped[col]
    newcolumn = [coldict[l] for l in test_set_to_encode[:, col]]
    test_set_to_encode[:, col] = newcolumn
test_set.ix[:, 'x0':'x20'] = test_set_to_encode
test_matr = test_set.ix[:, 'x0':'x20'].as_matrix()
ml = len(matr)

matr = np.append(matr, test_matr, axis=1)

newdf = enc.fit_transform(matr)

mmm = {}
ndf = newdf.tocsr()
for i in range(newdf.shape[1]):
    mmm[str(i)] = ndf[:ml, i]

df_cat = pd.DataFrame(mmm, index=[0])
df.ix[:, 0:20] = df_cat

train_features = df.ix[:, 'x0':'x61'].fillna(0).as_matrix()
train_true = df['y'].tolist()

trtrfe = train_features[:35000, :]
trtrtrue = train_true[:35000]

trtefe = train_features[35000:, :]
trtetrue = train_true[35000:]


#encoding test set
mmm = {}
for i in range(newdf.shape[1]):
    mmm[str(i)] = ndf[ml:, i]

df_cat = pd.DataFrame(mmm, index=[0])
test_set.ix[:, 'x0':'x20'] = df_cat

test_features = test_set.ix[:, 'x0':'x61'].fillna(0).as_matrix()

best_score = 0
best_model = None

print("learning")

# for depth in [3, 5, 7, 10, 12, 15, 20, 30, 50, 70]:
#     for leaf_samples in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20, 40, 60, 150]:

# for subsample in [1, 0.9, 0.8, 0.7, 0.6]:
#     for colsample_bytree in [1, 0.9, 0.8, 0.7, 0.6]:
#         model = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05, nthread=4, subsample=subsample, colsample_bytree=colsample_bytree)
#         model.fit(trtrfe, trtrtrue)
#         # mean accuracy on the given test data and labels
#         predicted = model.predict(trtefe)
#         score = model.score(trtefe, trtetrue)
#         if score > best_score:
#             best_model = model
#             best_score = score
#         print(subsample, '\t', colsample_bytree, '\t', score)
#         # print(classification_report(trtetrue, predicted))
# best_model.fit(trtefe, trtetrue)

model = xgb.XGBClassifier(max_depth=150, n_estimators=450, learning_rate=0.05, nthread=4, subsample=0.7, colsample_bytree=0.7).fit(train_features, train_true)
predicted = model.predict(test_features)

res = {}
res["y"] = predicted
res["ID"] = [i for i in range(len(predicted))]
result = pd.DataFrame(res)
result.to_csv("data/sol.csv", columns = ['ID', 'y'], index=False)