import numpy as np
import pandas as pd
from sklearn import svm

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

feature_columns = ["create_age_months", "expiry_age_months", "update_age_days"]
X_train = train_data[feature_columns]
y_train = train_data["phishy"]
X_test = test_data[feature_columns]

clf = svm.SVC()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(predictions)
