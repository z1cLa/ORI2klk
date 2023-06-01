import random
import matplotlib.pyplot as plt

def fit(x, y):
    assert len(x) == len(y)
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    numerator = denominator = 0
    for i in range(n):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

def predict(x, slope, intercept):
    return intercept + slope * x

def make_predictions(x, slope, intercept):
    y_pred = [predict(xi, slope, intercept) for xi in x]
    return y_pred

if __name__ == '__main__':
    random.seed(1337)
    x = [xi for xi in range(50)]
    y = [(xi + random.randint(-5, 5)) for xi in x]
    slope, intercept = fit(x, y)
    y_pred = make_predictions(x, slope, intercept)
    plt.plot(x, y, '.')
    plt.plot(x, y_pred, 'b')
    plt.title(f'slope: {slope:.2f}, intercept: {intercept:.2f}')
    plt.show()
