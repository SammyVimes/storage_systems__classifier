from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics.classification import classification_report
import pandas as pd
__author__ = 'semyon'


print("reading")
csv = pd.read_csv("data/train.csv")

print("slicing")
train_features = csv.ix[:, 'x23':'x61'].fillna(0).as_matrix()
train_true = csv['y'].tolist()

trtrfe = train_features[:35000, :]
trtrtrue = train_true[:35000]

trtefe = train_features[35000:, :]
trtetrue = train_true[35000:]

print("learning")

for depth in [7, 10, 12, 15, 20, 30, 50, 70]:
    for leaf_samples in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20, 40, 60, 150]:
        # model = GradientBoostingClassifier(n_estimators=10, max_depth=depth, min_samples_leaf=leaf_samples, verbose=1)
        model = RandomForestClassifier(n_estimators=50, max_depth=depth, min_samples_leaf=leaf_samples, verbose=0,
                                       n_jobs=4)
        model.fit(trtrfe, trtrtrue)
        # mean accuracy on the given test data and labels
        # print depth, '\t', leaf_samples, '\t', model.score(trtefe, trtetrue)
        predicted = model.predict(trtefe)
        print(classification_report(trtetrue, predicted))