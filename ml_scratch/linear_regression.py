import numpy as np


class LinearRegression():
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.regression_line = None


    def fit(self, X_train, y_train):
        x_hat = X_train.mean()
        y_hat = y_train.mean()

        s_x = X_train.std()**2
        s_xy = np.cov(X_train, y_train)[0, 1]

        self.slope = s_xy / s_x
        self.intercept = y_hat - self.slope*x_hat

        self.regression_line = lambda x, slope, intercept: slope*x + intercept


    def predict(self, X_test):
        y_pred = [self.regression_line(x) for x in X_test]
        return y_pred
