# -*- coding: utf-8 -*-
import numpy as np
from numpy.core.numeric import NaN
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dok import dok_matrix
import time
import math

import pandas as pd
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.manifold.t_sne import TSNE
from sklearn.metrics.classification import classification_report, confusion_matrix
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing.imputation import Imputer

import sys

#хак для винды
sys.path.append("C:\\myFiles\\builtdir\\xgboost\\python-package")

import xgboost as xgb

print("start")
print("reading train")

train_df = pd.read_csv("data/train.csv")
# train_df.iloc[np.random.permutation(len(train_df))]
train_df.reset_index(drop=True)

CAT_COUNT = 18


# меняем колонки, чтобы категориальные были в начале
def reorder_cols(df):
    if 'ID' in df.columns:
        print("removing ID")
        df.pop('ID')
    # if 'y' in df.columns:
    #     print("removing y")
    #     df.pop('y')
    df.pop("x18")
    df.pop("x21")
    df.pop("x10")
    cols = list(df.columns)
    # df.insert(1, 'x21', df.pop('x21'))
    df.insert(1, 'x22', df.pop('x22'))
    df.insert(len(cols) - 1, 'x8', df.pop('x8'))
    df.insert(len(cols) - 1, 'x13', df.pop('x13'))
    return df


def build_col_dict(column, column2):
    labels = set()
    for label in column:
        labels.add(label)
    for label in column2:
        labels.add(label)
    final_dict = {y: x for x, y in enumerate(list(labels))}
    final_dict[np.nan] = NaN
    return final_dict


print("reoredering cols")

train_df = reorder_cols(train_df)
# train_df = train_df.dropna()
train_true = train_df['y'].tolist()
train_df.pop('y')
print(train_df.columns)

print("convert cats to matrix")

train_matrix = train_df.ix[:, 0:CAT_COUNT].as_matrix()
rows, columns = train_matrix.shape

mapped = []
dicts = []

print("reading test")

test_df = pd.read_csv("data/test.csv")
test_df.reset_index(drop=True)
test_df = reorder_cols(test_df)
test_set_to_encode = test_df.ix[:, 0:CAT_COUNT].as_matrix()

print("Building dicts for encoding")

# для каждой колонки набираем словарь

for col in range(CAT_COUNT):
    coldict = build_col_dict(train_matrix[:, col], test_set_to_encode[:, col])
    mapped.append(coldict)
    print("column " + str(test_df.columns[col]) + " dict size = " + str(len(coldict)) + " dict " + str(coldict))

for col in range(CAT_COUNT):
    coldict = mapped[col]
    newcolumn = [coldict[l] for l in train_matrix[:, col]]
    train_matrix[:, col] = newcolumn
    newcolumn = [coldict[l] for l in test_set_to_encode[:, col]]
    test_set_to_encode[:, col] = newcolumn

train_df.ix[:, 0:CAT_COUNT] = train_matrix

print("building extra mult features")


def build_extra_features(noncat_matrix):
    X = dok_matrix((noncat_matrix.shape[0], noncat_matrix.shape[1] * 2))
    xs, ys = noncat_matrix.nonzero()
    print(len(xs), "nonzero elems")
    count = 0
    for x, y in zip(xs, ys):
        count += 1
        if count % 1000 == 0:
            print(count)
        val = noncat_matrix[x, y]
        if val - math.floor(val) != 0.0:
            for i in range(20):
                if abs(abs(val) * i - math.ceil(abs(val) * i)) < 0.001:
                    X[x, 2 * y] = math.ceil(abs(val) * i)
                    X[x, 2 * y + 1] = i
    return X


# категории
print("building train")
train_cat_matr = train_df.ix[:, 0:CAT_COUNT].as_matrix()
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
train_cat_matr = imp.fit_transform(train_cat_matr)
# imp2 = Imputer(missing_values='NaN', strategy='median')
train_noncat_matr = train_df.ix[:, CAT_COUNT:].fillna(0).as_matrix()
# train_noncat_matr = train_df.ix[:, CAT_COUNT:].as_matrix()
# train_noncat_matr = imp2.fit_transform(train_noncat_matr)
# allf = np.hstack((train_cat_matr, train_noncat_matr))


print("building test")
test_df.ix[:, 0:CAT_COUNT] = test_set_to_encode
test_cat_matr = test_df.ix[:, 0:CAT_COUNT].as_matrix()
test_cat_matr = imp.transform(test_cat_matr)
test_noncat_matr = test_df.ix[:, CAT_COUNT:].fillna(0).as_matrix()
# test_noncat_matr = test_df.ix[:, CAT_COUNT:].as_matrix()
# test_noncat_matr = imp2.transform(test_noncat_matr)
# test_extra_matr = build_extra_features(test_noncat_matr[:,:10])
# test_noncat_matr = np.hstack((test_noncat_matr, test_extra_matr))

print("One-hot-encoding")

enc = OneHotEncoder(categorical_features=range(CAT_COUNT))
preprocessed_features = np.hstack((train_cat_matr, train_noncat_matr))

enc_train_df = enc.fit_transform(preprocessed_features)

print("test")
enc_test_df = enc.transform(np.hstack((test_cat_matr, test_noncat_matr)))

train_features = csr_matrix(enc_train_df)
test_features = csr_matrix(enc_test_df)

print("Transformed")

print("train_features shape: ", train_features.shape)

trtrfe = train_features[:20000, :].todense()
trtrtrue = train_true[:20000]

trtefe = train_features[20000:, :].todense()
trtetrue = train_true[20000:]

print("Building test set")

best_score = 0
best_model = None


def write_sol(predicted, fname):
    res = {}
    res["y"] = predicted
    res["ID"] = [i for i in range(len(predicted))]
    result = pd.DataFrame(res)
    print(fname)
    result.to_csv(fname, columns=['ID', 'y'], index=False)


for n_estimators in [100, 450, 550, 1000]:
    print("learning " + str(n_estimators) + "  estimators")
    for subsample in [0.65, 0.7, 0.8, 1]:
        for depth in [10, 340, 500]:
            for learning_rate in [0.05, 0.005, 0.5, 0.8]:
                print(learning_rate, depth, subsample,)
                model = xgb.XGBClassifier(max_depth=depth, n_estimators=n_estimators, learning_rate=learning_rate,
                                          nthread=2, subsample=subsample, silent=True, colsample_bytree=0.8)

                # model = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators, n_jobs=2, min_samples_leaf=3,
                #                                class_weight={0: 3, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 1})

                # model = LogisticRegression(C=subsample, verbose=0, penalty='l1', max_iter=100)
                # model = KNeighborsClassifier(n_neighbors=learning_rate)
                # model = xgb.XGBRegressor(max_depth=depth, n_estimators=n_estimators, learning_rate=learning_rate,
                #                          nthread=1, subsample=subsample, silent=True, colsample_bytree=0.8)
                # model = LinearSVC(C=0.9, penalty='l2', dual=False, verbose=1, max_iter=100000)

                model.fit(trtrfe, trtrtrue)
                # mean accuracy on the given test data and labels
                predicted = [math.floor(x) for x in model.predict(trtefe)]

                score = model.score(trtefe, trtetrue)
                print("score =", score)

                print(classification_report(trtetrue, predicted))
                print(confusion_matrix(trtetrue, predicted))

                if score > best_score or True:
                    best_model = model
                    best_score = score

                    best_model.fit(train_features, train_true)
                    predicted = [math.floor(x) for x in best_model.predict(test_features)]
                    fname = "data/net_result/sol_" + str(score) + "_" + str(time.time()) + ".csv"
                    write_sol(predicted, fname)
                    print("this model", depth, '\t', subsample, "\t", score)
                    print("best model", best_score)

best_model.fit(trtefe, trtetrue)
predicted = [math.floor(x) for x in best_model.predict(test_features)]
write_sol(predicted, "data/net_result/sol.csv")
print("done")