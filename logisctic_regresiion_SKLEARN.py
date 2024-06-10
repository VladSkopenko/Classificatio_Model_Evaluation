from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=1000).fit(X, y)
result = clf.score(X, y)
print(result)
clss = clf.predict(X)  # значення
y_predict = clf.predict_proba(X)  # ймовірності

print(log_loss(y, y_predict))
