import numpy as np
import matplotlib.pyplot as plt
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
