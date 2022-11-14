import math
import numpy as np

from ft_progress import ft_progress

class MyLinearRegressionException(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, progress_bar=False):
        if (not isinstance(alpha, float) and not isinstance(alpha, int)) or alpha <= 0:
            raise MyLinearRegressionException("MyLinearRegressionException: Alpha must be a float > 0")
        self.alpha = alpha
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise MyLinearRegressionException("MyLinearRegressionException: max_iter must be an int > 0")
        self.max_iter = max_iter
        if isinstance(progress_bar, bool):
            self.progress_bar = progress_bar
        else:
            self.progress_bar = False
        if not isinstance(thetas, np.ndarray):
            raise MyLinearRegressionException("MyLinearRegressionException: Bad thetas")
        if len(thetas) == 0:
            raise MyLinearRegressionException("MyLinearRegressionException: Bad thetas")
        if thetas.shape != (2, 1):
            raise MyLinearRegressionException("MyLinearRegressionException: Bad thetas")
        self.thetas = thetas

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y and update thetas
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        Returns:
            None
        """
        if not isinstance(x,np.ndarray) or not isinstance(y, np.ndarray):
            print("Error: x or y are not good Numpy.ndarray.")
            return 
        if len(x) == 0 or len(y) == 0:
            print("Error: x or y are empty.")
            return 
        if (len(x.shape) > 1 and x.shape[1] != 1) or (len(y.shape) > 1 and y.shape[1] != 1):
            print("Error: x or y have got bad shape.")
            return 
        
        if x.shape != y.shape:
            print("Error: x and y have not the same shape.")
            return 
    
        list = range(self.max_iter)
        if self.progress_bar:
            list = ft_progress(list)
        
        mse_list = []
        for i in list:
            m = len(x)
            h = self.predict_(x)
            diff = h - y
            gradien = np.array([[diff.sum() / m], [(diff * x).sum() / m]])
            t0 = self.thetas[0][0]
            t1 = self.thetas[1][0]
            t0 -= (self.alpha * gradien[0][0])
            t1 -= (self.alpha * gradien[1][0])
            self.thetas= np.array([t0, t1]).reshape((-1, 1))
            y_hat = self.predict_(x)
            mse_list.append(MyLinearRegression.mse_(y, y_hat))
        return mse_list

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
        """
        if not isinstance(x,np.ndarray):
            return None
        if len(x) == 0:
            return None
        if (len(x.shape) > 1 and x.shape[1] != 1):
            return None
        x_1 = np.c_[np.ones(x.shape[0]), x]
        return x_1.dot(self.thetas)
    
    def loss_elem_(self, y, y_hat):
        """
        Description:  Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)\
            or y.shape[-1] == 0 or y_hat.shape[-1] == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        try:
            return((y_hat - y) * (y_hat - y))
        except Exception:
            return None


    def loss_(self, y, y_hat):
        """
        Description: Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)\
            or y.shape[-1] == 0 or y_hat.shape[-1] == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        try:
            return self.loss_elem_(y, y_hat).sum() / (2 * len(y))
        except Exception:
            return None
#****************************************************************
# Class' Methods
#****************************************************************
    def mse_(y, y_hat):
        """
        Description:
            Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            loss_elem = (y_hat - y) * (y_hat - y)
            return loss_elem.sum() / len(y)
        except Exception:
            return None
    
    def rmse_(y, y_hat):
        """
        Description:
            Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            return math.sqrt(MyLinearRegression.mse_(y, y_hat))
        except Exception:
            return None
