from numpy import (abs, array, sqrt, log)
import math


def compute_skew(arr):
    """_summary_

    Args:
        y (_type_): _description_

    Utils
    - https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics
    - https://www.thoughtco.com/what-is-skewness-in-statistics-3126242

    Returns:
        _type_: _description_
    """
    arr = array(arr).flatten()
    n = arr.size
    const = n * sqrt(n - 1) / (n - 2)
    y_m = arr.mean()
    num = ((arr - y_m)**3).sum()
    den = (((arr - y_m)**2).sum())**(3 / 2)
    return const * num / den


def compute_kurtosis(arr, residuals=None):
    """_summary_

    Args:
        y (list|array-like): _description_

    Utils
    - https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics

    Returns:
        _type_: _description_
    """
    arr = array(arr).flatten()
    n = arr.size
    const = (n - 1) * n * (n + 1) / ((n - 2) * (n - 3))
    y_m = arr.mean()
    num = ((arr - y_m)**4).sum()
    den = (((arr - y_m)**2).sum())**(4 / 2)
    assert num.shape == ()
    assert den.shape == ()
    assert y_m.shape == ()
    const2 = 3 * ((n - 1)**2) / ((n - 2) * (n - 3))
    return (const * num / den) - const2


def compute_aic_bic(dfr: int, n: int, llh: float, method: str = "basic"):
    """_summary_

    Utils
        - It adds a penalty that increases the error when including additional terms. The lower the AIC, the better the model.
        - https://medium.com/analytics-vidhya/probabilistic-model-selection-with-aic-bic-in-python-f8471d6add32

    Args:
        dfr (int): nb_predictors(not including the intercept)
        dfe (int): nb of observations
        llh (float): log likelihood

    Question
        what about mixed models ?

    Returns:
        float: aicself, y_true, y_pred
    """
    K = dfr  # number of independent variables to build model==nb_predictors(not including the intercept)
    m1, m2 = 2, log(n)
    aic = m1 * K - 2 * llh
    bic = m2 * K - 2 * llh

    if method == "basic":
        return aic + m1, bic + m2
    elif method == "log":
        return aic, bic
    elif method == "correct":
        return aic + 2 * K * (K + 1) / (n - 1 - K), bic
    


class PredictionMetrics:

    def __init__(self, y_true: list, y_pred: list, binary=False) -> None:
        self.y_true = y_true 
        self.y_pred = y_pred
        self.binary = bool(binary)
        if binary:
            self.y_true_bin = (y_true>0.5).astype('int')
            self.y_pred_bin = (y_pred>0.5).astype('int')
            self.confusion_matrix = None

    def compute_mae(self):
        y = array(self.y_true)
        y_pred = array(self.y_pred)
        assert y.shape == y_pred.shape
        return abs(y - y_pred).mean()

    
    def compute_log_likelihood(self, std_eval: float, debug=False):
        if self.binary:
            return self._log_likelihood_logit()
        else:
            return self._log_likelihood_lin_reg(std_eval=std_eval, debug=debug)
    
    def _log_likelihood_lin_reg(self, std_eval: float, debug=False):
        """_summary_

        Args:
            y (list): _description_
            self.y_pred (list): _description_
            std_eval (float): _description_
            debug (bool, optional): _description_. Defaults to False.
        Utils
            - https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/mle-regression.pdf
        Returns:
            log_likelihood: _description_
        """
        assert self.binary == False
        y = self.y_true
        y_pred = self.y_pred
        y = array(y, dtype=float)
        y_pred = array(y_pred, dtype=float)
        sigma = float(std_eval)
        assert y.ndim == 1
        assert y_pred.ndim == 1
        assert len(y) == len(y_pred)
        n = len(y)
        sigma_carre = sigma**2
        CST = math.log(2 * math.pi * sigma_carre)
        SST = ((y - y_pred)**2).sum()
        log_likelihood = -(n / 2) * CST - SST / (2 * sigma_carre)
        return log_likelihood
    
    def _log_likelihood_logit(self):
        assert self.binary == True
        y = self.y_true
        yp = self.y_pred
        return (y * log(yp) + (1 - y) * log(1 - yp)).sum()



    def get_confusion_matrix(self):
        assert self.binary == True
        if self.confusion_matrix is not None: return self.confusion_matrix
        _pred_neg = self.y_pred_bin[self.y_true_bin==0]
        _pred_pos = self.y_pred_bin[self.y_true_bin==1]
        tn, fp, fn ,tp = (_pred_neg==0).sum(), (_pred_pos==0).sum(), (_pred_neg==1).sum(), (_pred_pos==1).sum()
        self.confusion_matrix = [[tn, fp], [fn ,tp]]
        return self.confusion_matrix

    def get_binary_accuracy(self):
        assert self.binary == True
        #return abs(self.y_true_bin==self.y_pred_bin).sum()/len(self.y_true_bin)
        (tn, fp), (fn ,tp) = self.get_confusion_matrix()
        return (tp+tn)/(tp+tn+fp+fn)

    def get_precision_score(self):
        assert self.binary == True
        (tn, fp), (fn ,tp) = self.get_confusion_matrix()
        return tp/(tp+fp)
    
    def get_recall_score(self):
        assert self.binary == True
        (tn, fp), (fn ,tp) = self.get_confusion_matrix()
        return tp/(tp+fn)

    def get_f1_score(self):
        assert self.binary == True
        prec = self.get_precision_score() 
        rec = self.get_recall_score()
        return 2*(prec*rec)/(prec+rec)
    