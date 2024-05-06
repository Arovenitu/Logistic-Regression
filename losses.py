import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:

    def func(self, X, y, w):
        deg = X.dot(w[1:]) + w[0]
        return np.mean(logsumexp(np.vstack((np.zeros(X.shape[0]),-y * deg)).T,axis = 1))

    def grad(self, X, y, w):
        X_0 = np.hstack((np.ones((X.shape[0],1)),X))
        deg = X.dot(w[1:]) + w[0]
        return np.mean(((-y * X_0.T) * expit(-y*deg)).T, axis=0)


class BinaryLogisticLoss(BaseLoss):

    def __init__(self, l2_coef):
        self.l2_coef = l2_coef
        self.l2_func = l2_coef

    def func(self, X, y, w):
        return super().func(X, y, w) + self.l2_func * (np.linalg.norm(w[1:],2) ** 2)
    
    def func_0(self, X, y, w):
        return super().func(X, y, w)    

    def grad(self, X, y, w):
        w_ = w.copy()
        w_[0] = 0
        return super().grad(X, y, w) + 2 * self.l2_coef * w_

