import sklearn
import pandas as pd
import random
import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
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


# enc = OneHotEncoder(categorical_features=range(20), sparse=False)
# matr = df.ix[:, 0:20].as_matrix()

rows, columns = test_set_to_encode.shape
for col in range(columns):
    coldict = mapped[col]
    newcolumn = [coldict[l] for l in test_set_to_encode[:, col]]
    test_set_to_encode[:, col] = newcolumn
test_set.ix[:, 'x0':'x20'] = test_set_to_encode
test_matr = test_set.ix[:, 'x0':'x20'].as_matrix()
# ml = len(matr)

# matr = np.append(matr, test_matr, axis=1)

# newdf = enc.fit_transform(matr)

# mmm = {}
# for i in range(newdf.shape[1]):
#     mmm[str(i)] = newdf[:ml, i]

# df_cat = pd.DataFrame(mmm)
# df = pd.concat([df_cat, df.ix[:, 21:]], axis=1)

train_features = df.ix[:, :df.shape[1] - 2].fillna(0).as_matrix()
train_true = df['y'].tolist()

trtrfe = train_features[:35000, :]
trtrtrue = train_true[:35000]

trtefe = train_features[35000:, :]
trtetrue = train_true[35000:]


# mmm = {}
# for i in range(newdf.shape[1]):
#     mmm[str(i)] = newdf[:ml, i]
#
# df_cat = pd.DataFrame(mmm)
# test_set = pd.concat([df_cat, test_set.ix[:, 22:]], axis=1)

test_features = test_set.ix[:, :].fillna(0).as_matrix()

best_score = 0
best_model = None

print("learning")

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict_proba(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(X))
        return np.mean(self.predictions_, axis=0)

    def predict(self, X):
        self._predictions_ = list()
        for classifier in self.classifiers:
            self._predictions_.append(classifier.predict(X))
        return np.median(self._predictions_, axis=0)

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
one_vs_rest_model = classifier.fit(train_features, train_true)
predict_proba = one_vs_rest_model.predict_proba(test_features)


model = xgb.XGBClassifier(max_depth=3, n_estimators=10, learning_rate=0.05, nthread=4, subsample=0.7, colsample_bytree=0.7).fit(train_features, train_true)
predicted = model.predict(test_features)
for row in range(predict_proba.shape[0]):
    xgb_predicted = predicted[row]
    predicted[row] = np.arggcount((predict_proba[row], xgb_predicted))

res = {}
res["y"] = predicted
res["ID"] = [i for i in range(len(predicted))]
result = pd.DataFrame(res)
result.to_csv("data/sol.csv", columns = ['ID', 'y'], index=False)