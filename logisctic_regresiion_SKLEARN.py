from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


X, Y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=1000).fit(X, Y)
result = clf.score(X, Y)
print(result)
