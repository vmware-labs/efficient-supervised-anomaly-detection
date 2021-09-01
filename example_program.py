# Copyright 2018-2021 VMware, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from rade_classifier import RadeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import time

X, y = make_classification(n_samples=1000000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False, weights=[0.99, 0.01])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# RADE
rade_clf = RadeClassifier()

start = time.time()
rade_clf.fit(X_train, y_train)
end = time.time()
rade_train_time = end - start
print("RADE training time is: {:0.2f} seconds".format(rade_train_time))

start = time.time()
rade_y_predicted = rade_clf.predict(X_test)
end = time.time()
rade_predict_time = end - start
print("RADE prediction time is: {:0.2f} seconds".format(rade_predict_time))

rade_f1 = f1_score(y_test, rade_y_predicted, average='macro')
print("RADE classification_report:")
print(classification_report(y_test, rade_y_predicted, digits=5))

# Random Forest
rf_clf = RandomForestClassifier()

start = time.time()
rf_clf.fit(X_train, y_train)
end = time.time()
rf_train_time = end - start
print("Random Forest training time is: {:0.2f} seconds".format(rf_train_time))

start = time.time()
rf_y_predicted = rf_clf.predict(X_test)
end = time.time()
rf_predict_time = end - start
print("Random Forest prediction time is: {:0.2f} seconds".format(rf_predict_time))

rf_f1 = f1_score(y_test, rf_y_predicted, average='macro')
print("Random Forest classification_report:")
print(classification_report(y_test, rf_y_predicted, digits=5))

# XGBoost
xgb_clf = XGBClassifier()

start = time.time()
xgb_clf.fit(X_train, y_train)
end = time.time()
xgb_train_time = end - start
print("XGBoost training time is: {:0.2f} seconds".format(xgb_train_time))

start = time.time()
xgb_y_predicted = xgb_clf.predict(X_test)
end = time.time()
xgb_predict_time = end - start
print("XGBoost prediction time is: {:0.2f} seconds".format(xgb_predict_time))

xgb_f1 = f1_score(y_test, xgb_y_predicted, average='macro')
print("XGBoost classification_report:")
print(classification_report(y_test, xgb_y_predicted, digits=5))

print('RADE vs. Random Forest and XGBoost:')
print ('Training time: RADE is {:0.1f}x faster than Random Forest, and {:0.1f}x faster than XGBoost.'.
    format(rf_train_time / rade_train_time, xgb_train_time / rade_train_time))
print ('Prediction time: RADE is {:0.1f}x faster than Random Forest, and {:0.1f}x faster than XGBoost.'.
    format(rf_predict_time / rade_predict_time, xgb_predict_time / rade_predict_time))
print ('Macro F1: RADE is better by {:+0.5f} as compared to Random Forest, and by {:+0.5f} as compared to XGBoost.'.
    format(rade_f1 - rf_f1, rade_f1 - xgb_f1))
