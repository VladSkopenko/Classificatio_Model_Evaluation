import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print(mean_squared_error(np.array([4]), np.array([5])), mean_squared_error(np.array([4]), np.array([3])))
print(mean_squared_error(np.array([4]), np.array([4])))
print(mean_squared_error(np.array([4]), np.array([50])), mean_absolute_error(np.array([4]), np.array([50])))


def custom_loss(y_true, y_pred, t):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2) * t, (residual ** 2) * (1 - t))
    return np.mean(loss)


print(custom_loss(np.array([4]), np.array([5]), t=0.5), custom_loss(np.array([4]), np.array([4]), t=0.5),
      custom_loss(np.array([4]), np.array([3]), t=0.5))

print(custom_loss(np.array([4]), np.array([5]), t=0.25), custom_loss(np.array([4]), np.array([4]), t=0.25),
      custom_loss(np.array([4]), np.array([3]), t=0.25))

print(custom_loss(np.array([4]), np.array([5]), t=0.75), custom_loss(np.array([4]), np.array([4]), t=0.75),
      custom_loss(np.array([4]), np.array([3]), t=0.75))

y_true = np.linspace(0, 100, 50)

y_pred = {}
noise = np.random.normal(0, 1, 50)
for c_std in np.linspace(0, 99, 100, dtype=int):
    y_pred[c_std] = y_true + c_std * noise

for func in (mean_squared_error, mean_absolute_error, r2_score):
    plt.plot([func(y_true, y_pred[i]) for i in y_pred], label=func.__name__)
    plt.xlabel('std')
    plt.ylabel(func.__name__)
    plt.legend()
    plt.show()
