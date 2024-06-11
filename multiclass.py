import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from logisctic_regresiion_SKLEARN import x, y

ovo_clf = OneVsOneClassifier(LogisticRegression()).fit(x, y) 

a = np.column_stack([ovo_clf.predict(x), y])
print(a)



