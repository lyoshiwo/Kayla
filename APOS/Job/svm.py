from sklearn.svm import LinearSVC
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import time

print time.localtime()
[x, y, _, _] = pd.read_pickle('pickle/manual_position_size_salary.pkl')
count = 0
for i in range(x.shape[1]):
    count += x[:, i].max() + 1
    x[:, i] += count
enc = preprocessing.OneHotEncoder()
enc.fit(x)
x = enc.transform(x)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=713)
clf = LinearSVC(penalty='l1', dual=False, verbose=1, max_iter=2000)
clf.fit(X_train, Y_train)
p_test = clf.predict(X_test)
# 1000, 0.392768
print metrics.accuracy_score(p_test, Y_test)
print time.localtime()
