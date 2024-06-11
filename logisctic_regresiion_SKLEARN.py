from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

x, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=1000).fit(x, y)
result = clf.score(x, y)
print(result)
clss = clf.predict(x)  # значення
y_predict = clf.predict_proba(x)  # ймовірності

print(log_loss(y, y_predict))
