import numpy as np
from sklearn.metrics import log_loss
"""
Cross entropy loss function
"""


def cross_entropy(targets, predictions):
    if len(targets) != len(predictions):
        print("Length of targets and predictions should be same")
    else:
        losses = []
        for yt, yp in zip(targets, predictions):
            first_part = yt * np.log(yp)
            second_part = (1 - yt) * np.log(1 - yp)
            merge = -1 * (first_part + second_part)
            losses.append(merge)
        return np.sum(losses) / len(targets)


def cross_entropy_m(targets, predictions):
    if len(targets) != len(predictions):
        print("Length of targets and predictions should be same")
    else:
        m = len(targets)
        first_part = np.dot(targets.T, np.log(predictions))
        second_part = np.dot((1 - targets).T, np.log(1 - predictions))
        ce = -(first_part + second_part) / m
        return ce


args = (np.array([0, 1, 0, 1, 1]),
        np.array([0.001, 0.99, 0.001, 0.99, 0.99]))

print("Приклад позитивних значень функції log loss  який свідчить про добре навчану модель")
print(cross_entropy(*args))
print(cross_entropy_m(*args))
print(log_loss(*args))
print("===========================")
args = (np.array([0, 1, 0, 1, 1]),
        np.array([0.99, 0.001, 0.99, 0.001, 0.001]))

print("Погано навчана модель внизу, але її легко  виправити якщо  перевернути значення")
print(cross_entropy(*args))
print(cross_entropy_m(*args))
print(log_loss(*args))

print("Найгірші результати це середені, воно свідчать що модель не навчалась нічому")