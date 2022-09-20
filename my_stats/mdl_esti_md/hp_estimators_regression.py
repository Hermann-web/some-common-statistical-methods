'''
todo
- refactor output (last lines)
- use "alternative" instead of "tail"
- use kwargs format while calling functions
- reorder fcts attributes
- Que signifie le R au carré négatif?: 
    - selon ma def, c'est  entre 0 et 1 à cause d'une somme mais c'est faux ??
    - https://qastack.fr/stats/183265/what-does-negative-r-squared-mean
'''

import os.path
import sys
from my_stats.utils_md.refactoring import RegressionFisherTestData
from my_stats.utils_md.compute_ppf_and_p_value import (
    get_p_value_f_test, )
from my_stats.utils_md.constants import COMMON_ALPHA_FOR_HYPH_TEST
from my_stats.hyp_vali_md.constraints import (check_zero_to_one_constraint,
                                              check_hyp_min_sample)
from numpy import (abs, random, array, sqrt, log)
import warnings
import math

print('mdl_esti_md.hp_estimator_regression: import start...')

sys.path.append(os.path.abspath("."))

# data manipulation and testing

SUM = sum

random.seed(233)

# hyp_validation

# utils

print('mdl_esti_md.hp_estimator_regression: ---import ended---')


def HPE_REGRESSION_FISHER_TEST(y: list,
                               y_hat: list,
                               nb_param: int,
                               alpha: float = COMMON_ALPHA_FOR_HYPH_TEST):
    """check if mean is equal accross many samples

    Args 
        y (list): array-like of 1 dim
        y_hat (list): array-like of 1 dim
        nb_param (int): number of parameter in the regression (include the intercept). ex: for 6 independant variables, nb_params=7
        alpha (float, optional): _description_. Defaults to COMMON_ALPHA_FOR_HYPH_TEST.

    Hypothesis
        H0: β1 = β2 = ... = βk-1 = 0; k=nb_params
        H1: βj ≠ 0, for at least one value of j

    Hypothesis
        - each sample is 
            - simple random 
            - normal
            - indepebdant from others
        - same variance 
            - attention: use levene test (plus robuste que fusher ou bartlett face à la non-normalité de la donnée)(https://fr.wikipedia.org/wiki/Test_de_Bartlett)


    Fisher test 
        - The F Distribution is also called the Snedecor’s F, Fisher’s F or the Fisher–Snedecor distribution
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
        - https://blog.minitab.com/fr/comprendre-lanalyse-de-la-variance-anova-et-le-test-f
        - http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm

    Returns:
        data: (RegressionFisherTestData)
    """

    check_zero_to_one_constraint(alpha)
    y = array(y, dtype=float)
    y_hat = array(y_hat, dtype=float)
    assert y.ndim == 1
    assert y_hat.ndim == 1
    assert len(y) == len(y_hat)
    n = len(y)
    k = int(nb_param)
    assert nb_param < n
    check_hyp_min_sample(n)

    SSR = ((y_hat - y.mean())**2).sum()  # explained by regression
    SSE = ((y - y_hat)**2).sum()  # SE (like MSE) of the model
    SST = SSE + SSR  # total error = ((y - y.mean())**2).sum()

    dfe = n - k  # error in sample # Degrees of Freedom for Error
    dfr = k - 1  # explained by the diff bewtwenn samples # Corrected Degrees of Freedom for Model
    dft = n - 1  # total # Corrected Degrees of Freedom Total

    MSR = SSR / dfr  # variance explained by the regresser # Mean of Squares for Model
    MSE = SSE / dfe  # variance within sample # Mean of Squares for Error
    # total variance = ( E(X**2) - E(X)**2 )/ (n-1) # Mean of Squares for Error
    MST = SST / dft

    #MSM = SSM / DFM

    R_carre = 1 - SSE / SST  # explained/total # 1-R_carre = SSE/SST
    assert R_carre <= 1 and R_carre >= 0
    R_carre_adj = 1 - MSE / MST  # 1-R_carre = MSE/MST
    if R_carre_adj < 0:
        warnings.warn(
            f"R_adj is negative ({round(R_carre_adj,2)}). your model is bad")
    F = MSR / MSE  # (explained variance) / (unexplained variance)
    # plus F est grand, plus la diff des mean s'explique par la difference entre les groupes
    # plus F est petit, plus la diff des mean s'explique par la variabilite naturelle des samples

    print(f" n = {n}  k = {k}")
    print(f"dfr= k-1 ={dfr}  dfe= n-k = {dfe}  dft= n-1 = {dft}")
    print(f"SSR={SSR} SSE={SSE} SST={SST} ")
    print(f"MSR={MSR} MSE={MSE} MST={MST} ")
    print(f"R_carre={R_carre} R_adj={R_carre_adj} F={F} ")

    # On veut F grand donc on prends un right tail =>

    p_value = get_p_value_f_test(Z=F, dfn=dfr, dfd=dfe)

    # rejection #we want to be far away from the mean (how p_value= how great is (p_hat - p0) = how far p_hat is from p0 considered as mean)
    reject_null = True if p_value < alpha else False
    return RegressionFisherTestData(DFE=dfe,
                                    SSE=SSE,
                                    MSE=MSE,
                                    DFR=dfr,
                                    SSR=SSR,
                                    MSR=MSR,
                                    DFT=dft,
                                    SST=SST,
                                    MST=MST,
                                    R_carre=R_carre,
                                    R_carre_adj=R_carre_adj,
                                    F_stat=F,
                                    p_value=p_value,
                                    reject_null=reject_null)


def compute_log_likelihood(y: list, y_hat: list, std_eval: float, debug=False):
    """_summary_

    Args:
        y (list): _description_
        y_hat (list): _description_
        std_eval (float): _description_
        debug (bool, optional): _description_. Defaults to False.
    Utils
        - https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/mle-regression.pdf
    Returns:
        log_likelihood: _description_
    """
    y = array(y, dtype=float)
    y_hat = array(y_hat, dtype=float)
    sigma = float(std_eval)
    assert y.ndim == 1
    assert y_hat.ndim == 1
    assert len(y) == len(y_hat)
    n = len(y)
    sigma_carre = sigma**2
    CST = math.log(2 * math.pi * sigma_carre)
    SST = ((y - y_hat)**2).sum()
    log_likelihood = -(n / 2) * CST - SST / (2 * sigma_carre)
    return log_likelihood


def compute_mae(y, y_hat):
    y = array(y)
    y_hat = array(y_hat)
    assert y.shape == y_hat.shape
    return abs(y - y_hat).mean()


def compute_skew(y: list):
    """_summary_

    Args:
        y (_type_): _description_

    Utils
    - https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics
    - https://www.thoughtco.com/what-is-skewness-in-statistics-3126242

    Returns:
        _type_: _description_
    """
    y = array(y).flatten()
    n = y.size
    const = n * sqrt(n - 1) / (n - 2)
    y_m = y.mean()
    num = ((y - y_m)**3).sum()
    den = (((y - y_m)**2).sum())**(3 / 2)
    return const * num / den


def compute_kurtosis(y):
    """_summary_

    Args:
        y (list|array-like): _description_

    Utils
    - https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics

    Returns:
        _type_: _description_
    """
    y = array(y).flatten()
    n = y.size
    const = (n - 1) * n * (n + 1) / ((n - 2) * (n - 3))
    y_m = y.mean()
    num = ((y - y_m)**4).sum()
    den = (((y - y_m)**2).sum())**(4 / 2)
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
        float: aic
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


if __name__ == "__main__":
    pass
else:
    print = lambda *args: ""
