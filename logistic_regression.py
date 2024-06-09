import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Сігмоїда """
    return 1 / (1 + np.exp(-x))


xx = np.linspace(0, 100, 101)
yy = sigmoid((xx - 60) / 5)
plt.plot(xx, yy)
plt.ylabel("Ймовірність складання іспиту")
plt.xlabel("Оцінка  на екзамені")
plt.title("Сігмоїдна функція")
plt.grid(True)
plt.show()

x = [-1, -1, 3]
w = [0.4, -0.5, 0.1]
res = sigmoid(np.dot(x, w))
print(res)
