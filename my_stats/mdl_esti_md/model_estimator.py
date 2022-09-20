'''
We know why t-student is useful
what about khi-2 ? we know 
fisher ? yes F

- add a fct to predict 
    - attention to extrapolation (unsern data) vs interpolation
- another for the curve showing the std
    - the interval should be narrower tinyer when X reacg the sample mean
- a good list of intel/reminder about the regression
    - https://sites.ualberta.ca/~lkgray/uploads/7/3/6/2/7362679/slides_-_multiplelinearregressionaic.pdf
    
'''

import os.path
import sys
from my_stats.mdl_esti_md.hp_estimators_regression import (
    HPE_REGRESSION_FISHER_TEST, compute_log_likelihood, compute_skew,
    compute_mae, compute_kurtosis, compute_aic_bic)
from my_stats.conf_inte_md.confidence_interval import IC_MEAN_ONE
from my_stats.hyp_vali_md.hypothesis_validator import (
    check_coefficients_non_zero, check_residuals_centered)
from my_stats.hyp_vali_md.constraints import (check_hyp_min_sample,
                                              check_sample_normality,
                                              check_zero_to_one_constraint)
from my_stats.utils_md.constants import COMMON_ALPHA_FOR_HYPH_TEST
from my_stats.utils_md.estimate_std import (compute_slope_std, estimate_std)
from my_stats.utils_md.preprocessing import (clear_list, clear_list_pair)
import warnings
from pandas import read_csv
from scipy.linalg import inv, det
from numpy import (random, array, zeros, power, dot, sqrt, diag)

print('mdl_esti_md.model_estimator: import start...')

sys.path.append(os.path.abspath("."))

# data manipulation
sum_loc = sum

random.seed(133)

# testing

# utils

# hyp_validation
# condidence_intervals

# regression metrics

print('mdl_esti_md.model_estimator: ---import end---')


def ME_Normal_dist(sample: list,
                   alpha=COMMON_ALPHA_FOR_HYPH_TEST,
                   debug=False):
    '''
    estimate a normal distribution from a sample

    visualisation: 
    - check if normal: 
        - sns.distplot(data.X)
        - check if qq-plot is linear #https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot
            ::from statsmodels.graphics.gofplots import qqplot 
            ::from matplotlib import pyplot
            ::qqplot(sample, line='s')
            ::pyplot.show()

    hypothesis 
    - X = m + N(0,s**2)

    - check normal hypothesis: https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

    lenght 
    - you may need data over 1000 samples to get
    '''
    check_zero_to_one_constraint(alpha)
    n = len(sample)
    check_hyp_min_sample(n)
    sample = clear_list(sample)

    # estimate mean
    # m = mean(sample) #estimate of the overall (marginal) mean
    data = IC_MEAN_ONE(confidence=0.95, sample=sample)
    m = data.parameter
    std_estimator = data.marginOfError

    # estimate std
    s = estimate_std(sample)  # e fl-> N(0,s**2)

    # check normality of residuals
    residuals = array(sample) - m
    passNormalitytest = check_sample_normality(residuals,
                                               alpha=alpha,
                                               debug=debug)

    if not passNormalitytest.testPassed:
        print('residuals does not look Gaussian (reject H0)')

    Testresults = {"residuals_normality": passNormalitytest}
    return m, std_estimator, s, Testresults


def ME_multiple_regression(X: list,
                           y: list,
                           degre: int,
                           debug=False,
                           alpha=COMMON_ALPHA_FOR_HYPH_TEST):
    '''
    estimate a regression model from two samples

    prediction
    - predict Y conditional on X, B, G, ... assuming that Y = pr[0] + pr[1]*X + pr[2]*B + pr[3]*G + N(0,s**2)
    - Y is a dependant variable
    - x, B, G, ...., s are independant ones => predictors of the dependant variables
    - If there is a time stamp of measures (or paired data), please add them as independant variables pr[0] + pr[4]*T1 +pr[5]*T2 + 
        => The correlation of the repeated measures needs to be taken into account, and time since administration needs to be added to the model as an independent variable.

    Questions of interest  
    - Are you interested in establishing a relationship?
    - Are you interested in which predictors are driving that relationship?

    visualisation: 
    - sns.scatterplot(X[i],y) for i in range(len(X)) 
    - check for Form_linear_or_not;Direction_pos_or_neg;Strengh_of_the_colinearity;Outliers

    hypothesis 
    - Y = pr[0] + pr[1]*X + pr[2]*B + pr[3]*G + err
    - err ~~> N(0,s**2)
    - variance(error)==s**2 is the same accross the data
    - var(Y/X)==s**2 ; E(Y/X) = pr[0] + pr[1]*X + pr[2]*B + pr[3]*G
    - pr[i] cst
    - pr[i] not null => i add a test hypothesis (to reject the null H0:coeff==0 against H1:coeff!=0), not a confidence interval (to check if 0 if not in)
    - non Collinearity a.k.a Multicollinearity
        - a correlation with be computed
        - Anyway, i does not change the predictive power not the efficieency of the model 
        - Too, i guess aic selection remove one right ?
        - But data about coefficients are not good because there is repetition
        - Regression Trees = can handle correlated data well

    prediction 
    - each pr[i] have a mean and a std based on normal distribution
    - Y too => 
        - Mean(Y) = y_hat = pr_h[0] + pr_h[1]*X + pr_h[2]*B + pr_h[3]*G
        - Some model can predict quantile(Y, 95%) but i will just add std(y_hat) later. uuh isn't s ?


    predictors 
    - pr[i], s**2

    lenght 
    - you may need data over 1000 samples to get

    Others 
    D'ont forget about the errors !
    Predictions have certain uncertainty => [ poorer fitted model => larger uncertainty]
    '''
    check_zero_to_one_constraint(alpha)
    pass


def ME_Regression(x: list,
                  y: list,
                  degre: int,
                  debug=False,
                  alpha=COMMON_ALPHA_FOR_HYPH_TEST):
    '''
    estimate a regression model from two samples

    prediction
    - predict Y conditional on X assuming that Y = pr[0] + pr[1]*X + pr[2]*X^2 + pr[3]*X^3 + N(0,s**2)
    - Y is a dependant variable
    - x, s are independant ones => predictors of the dependant variables
    - If there is a time stamp of measures (or paired data), please add them as independant variables pr[0] + var_exp_1*G + 


    visualisation: 
    - sns.scatterplot(X,Y) 

    hypothesis 
    - Y = pr[0] + pr[1]*X + pr[2]*X^2 + pr[3]*X^3 + err 
    - err ~~> N(0,s**2)
    - variance(error)==s**2 is the same accross the data
    - var(Y/X)==s**2 ; E(Y/X) = pr[0] + pr[1]*X + pr[2]*X^2 + pr[3]*X^3
    - pr[i] cst
    - pr[i] not null => i add a test hypothesis (to reject the null H0:coeff==0 against H1:coeff!=0), not a confidence interval (to check if 0 if not in)


    prediction 
    - each pr[i] have a mean and a std based on normal distribution
    - Y too => 
        - Mean(Y) = y_hat = pr_h[0] + pr_h[1]*X + pr_h[2]*X^2 + pr_[3]*X^3
        - Some model can predict quantile(Y, 95%) but i will just add std(y_hat) later. uuh isn't s ?


    predictors 
    - pr[i], s**2

    lenght 
    - you may need data over 1000 samples to get

    Others 
    D'ont forget about the errors !
    Predictions have certain uncertainty => [ poorer fitted model => larger uncertainty]

    utils
    - https://stats.stackexchange.com/questions/173271/what-exactly-is-the-standard-error-of-the-intercept-in-multiple-regression-analy
    '''

    check_zero_to_one_constraint(alpha)

    # reshape and remove nan
    x = array(x).flatten()
    y = array(y).flatten()
    x, y = clear_list_pair(x, y)
    # get sizes
    n = len(x)
    nb_param = int(degre) + 1
    # constraint
    check_hyp_min_sample(n)
    if n != len(y):
        raise Exception("x and y lenght must match")

    # create X
    X = zeros((n, nb_param))
    X[:, 0] = 1
    for i in range(1, nb_param):
        X[:, i] = power(x, i)
    assert X.shape == (n, nb_param)
    assert y.shape == (n, )

    # estimate coefficients
    b1 = dot(X.T, X)
    if det(b1) == 0:
        raise Exception("det==0")
    b1 = inv(b1)
    b2 = dot(X.T, y)
    coeffs = dot(b1, b2)
    assert coeffs.shape == (nb_param, )

    # compute residuals
    y_hat = dot(X, coeffs)  # y = y_hat + e
    residuals = y - y_hat
    assert residuals.shape == (n, )

    # compute standard error of the estimators
    # estimate standard deviation of the residual
    residu_std = estimate_std(residuals)  # e fl-> N(0,s**2)
    # estimate standard deviation of the coefficients
    assert b1.shape == (nb_param, nb_param)
    list_coeffs_std = residu_std * sqrt(
        diag(b1)
    )  # matrice de variance-covariance #les rzcine carre les elt diagonaux donnent les std
    assert list_coeffs_std.shape == (nb_param, )

    # test normality of the residuals
    passNormalitytest = check_sample_normality(residuals, alpha=alpha)
    if not passNormalitytest.testPassed:
        if debug:
            print('residuals does not look Gaussian (reject H0)')
    Testresults = {"residuals_normality": passNormalitytest}

    # test if mean != 0 for the residuals
    passed_residu_mean_null_test = check_residuals_centered(residuals,
                                                            alpha=alpha)
    if not passed_residu_mean_null_test.testPassed:
        if debug:
            print('residialss does not look centered')
    Testresults["residu_mean_null"] = passed_residu_mean_null_test

    # check if coefficients != 0
    nb_obs = n
    pass_non_zero_test = check_coefficients_non_zero(
        list_coeffs=coeffs,
        list_coeff_std=list_coeffs_std,
        nb_obs=nb_obs,
        alpha=alpha,
        debug=debug)
    if not passNormalitytest.testPassed:
        if debug:
            print('residuals does not look Gaussian (reject H0)')
    Testresults["coeff_non_zero"] = pass_non_zero_test

    # fisher test
    data = HPE_REGRESSION_FISHER_TEST(y=y, y_hat=y_hat, nb_param=nb_param)
    DFE, DFR = data.DFE, data.DFR
    SSE, MSE, SSR, MSR, SST, MST = data.SSE, data.MSE, data.SSR, data.MSR, data.SST, data.MST
    R_carre, R_carre_adj, F_stat, p_value = data.R_carre, data.R_carre_adj, data.F_stat, data.p_value
    pass_fisher_test = data.reject_null

    Testresults["significance"] = {
        "R_carre": R_carre,
        "R_carre_adj": R_carre_adj,
        "MSE": MSE,
        "DFE": DFE,
        "MSR": MSR,
        "DFR": DFR,
        "SSE": SSE,
        "SSR": SSR
    }
    Testresults["fisher_test"] = {
        "test_passed": pass_fisher_test,
        "F_stat": F_stat,
        "p_value": p_value
    }

    Testresults["metrics"] = {}
    log_likelihood = compute_log_likelihood(y=y,
                                            y_hat=y_hat,
                                            std_eval=residu_std)
    Testresults["metrics"]["log-likelihood"] = log_likelihood

    aic, bic = compute_aic_bic(dfr=DFR,
                               n=n,
                               llh=log_likelihood,
                               method="basic")
    Testresults["metrics"]["AIC"] = aic
    Testresults["metrics"]["BIC"] = bic

    # mse, rmse, mae
    Testresults["metrics"]["MSE"] = MSE
    MAE = compute_mae(y, y_hat)
    Testresults["metrics"]["MAE"] = MAE
    RMSE = sqrt(MSE)
    Testresults["metrics"]["RMSE"] = RMSE

    # skew, kurtosis
    Testresults["metrics"]["skew"] = compute_skew(y - y_hat)
    Testresults["metrics"]["kurtosis"] = compute_kurtosis(y - y_hat)

    return coeffs, list_coeffs_std, residu_std, Testresults


if __name__ == "__main__":
    pass
else:
    pass  # print = lambda *args: ""
