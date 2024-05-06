import numpy as np
from scipy.special import expit
import math
import time


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=100,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        indices = np.arange(X.shape[0])
        if w_0 is None:
            self.w = 0.05 * np.random.randn(X.shape[1]+1)
        else:
            self.w = w_0
        if X_val is None:
            X_val = X
            y_val = y
        f_ = self.loss_function.func(X,y,self.w)
        q = X.shape[0] // self.batch_size
        r = X.shape[0] % self.batch_size
        if trace == True:
            t = list()
            f = list()
            f_val = list()
            for i in range(1, self.max_iter + 1):
                np.random.shuffle(indices) #Перемешаем индексы
                start = time.time()
                f_last = f_
                for j in range(q):
                    epo_ind = indices[j * self.batch_size:(j+1)*self.batch_size] #какие индексы принимают участие в данной итерации
                    self.w -= (self.step_alpha/((i) ** self.step_beta)) * self.loss_function.grad(X[epo_ind],y[epo_ind],self.w)
                if r > 0:
                    self.w -= (self.step_alpha/((i) ** self.step_beta)) * self.loss_function.grad(X[indices[-r:]],y[indices[-r:]],self.w)
                f_ = self.loss_function.func(X,y,self.w)
                if i > 1:
                    if np.abs(f_- f_last) < self.tolerance:
                        break
                end = time.time()
                t.append(end - start)
                f.append(f_)   
                f_val.append(self.loss_function.func(X_val,y_val,self.w))
            history = dict()
            history['time'] = t
            history['func'] = f
            history['func_val'] = f_val
            return history
        else:
            for i in range(1, self.max_iter + 1):
                np.random.shuffle(indices) #перемешаем индексы
                f_last = f_
                for j in range(q):
                    epo_ind = indices[j * self.batch_size:(j+1)*self.batch_size] #какие индексы принимают участие в данной итерации
                    self.w -= (self.step_alpha/((i) ** self.step_beta)) * self.loss_function.grad(X[epo_ind],y[epo_ind],self.w)
                if r > 0:
                    self.w -= (self.step_alpha/((i) ** self.step_beta)) * self.loss_function.grad(X[indices[-r:]],y[indices[-r:]],self.w)
                f_ = self.loss_function.func(X,y,self.w)
                if np.abs(f_ - f_last) < self.tolerance:
                    break
    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        X_0 = np.hstack((np.ones((X.shape[0],1)),X))
        ans = X_0 @ self.w
        a_ = np.copy(ans)
        a_[np.where(ans > threshold)[0]] = 1
        a_[np.where(ans < threshold)[0]] = -1
        return a_
    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w[1:]

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func_0(X,y,self.w)

    def get_bias(self):
        """
        Get model bias

        Returns
        -------
        : float
            model bias
        """
        return self.w[0]
