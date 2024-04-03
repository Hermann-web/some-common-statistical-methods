# Statistical Analysis Module Documentation

This documentation provides an overview of the statistical analysis module, including its submodules, classes, and functions, featuring various utilities and functionalities for statistical analysis

## Module: `mdl_esti_md`

This module includes functionalities related to model estimation.

### Submodule: `prediction_metrics`

This submodule computes various prediction metrics.

- `compute_skew(arr)`: Computes skewness of an array.
- `compute_kurtosis(arr, residuals=None)`: Computes kurtosis of an array.
- `compute_aic_bic(dfr: int, n: int, llh: float, method: str = basic)`: Computes AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion).
- Class `PredictionMetrics`: Computes various prediction metrics for regression models.
  - `compute_mae(self)`: Computes Mean Absolute Error.
  - `compute_log_likelihood(self, std_eval: float = None, debug=False, min_tol: float = True)`: Computes log-likelihood.
  - `log_loss_flat(self, min_tol: float = None)`: Computes log loss.
  - `log_loss(self, min_tol: float = True)`: Computes log loss.
  - `get_confusion_matrix(self)`: Gets confusion matrix.
  - `get_binary_accuracy(self)`: Gets binary accuracy.
  - `get_precision_score(self)`: Gets precision score.
  - `get_recall_score(self)`: Gets recall score.
  - `get_f1_score(self)`: Gets F1 score.
  - `get_binary_regression_res(self)`: Gets binary regression results.

### Submodule: `__init__`

This submodule initializes the `mdl_esti_md` module.

### Submodule: `prediction_results`

This submodule handles regression result data and computations.

- Class `RegressionResultData`: Stores regression result data.
- `HPE_REGRESSION_FISHER_TEST(y: list, y_hat: list, nb_param: int, alpha: float = None)`: Performs regression Fisher test.
- `compute_linear_regression_results(crd: RegressionResultData, debug: bool = False)`: Computes linear regression results.
- `compute_logit_regression_results(crd: RegressionResultData, debug: bool = False)`: Computes logistic regression results.

### Submodule: `log_reg_example.py`

This submodule provides an example implementation of logistic regression.

- Class `LogisticRegression`: Implements logistic regression model.
- ...

### Submodule: `hp_estimators_regression.py`

This submodule provides functions for hypothesis testing in regression models.

- `sigmoid(z)`: Computes the sigmoid function.
- `log_loss(yp, y, min_tol: float = None)`: Computes log loss.
- Class `ComputeRegression`: Computes regression coefficients and results.
- ...

### Submodule: `model_estimator.py`

This submodule contains functions for model estimation.

- `ME_Normal_dist(sample: list, alpha=None, debug=False)`: Estimates a normal distribution from a sample.
- `ME_Regression(x: list, y: list, degre: int, logit=False, fit_intercept=True, debug=False, alpha: float = 0.05, nb_iter: int = 100000, learning_rate: float = 0.1)`: Performs regression analysis.
- `ME_multiple_regression(X: list, y: list, logit=False, fit_intercept=True, debug=False, alpha: float = 0.05, nb_iter: int = 100000, learning_rate: float = 0.1)`: Performs multiple regression analysis.
- `ME_logistic_regression(X: list, y: list, debug=False, alpha=None)`: Performs logistic regression analysis.

## Module: `hyp_vali_md`

This module handles hypothesis validation functionalities.

### Submodule: `constraints`

This submodule provides constraint checking functions.

- `check_zero_to_one_constraint(*args)`: Checks if values are between zero and one.
- `check_or_get_alpha_for_hyph_test(alpha=None)`: Checks or retrieves alpha value for hypothesis testing.
- `check_or_get_cf_for_conf_inte(confidence=None)`: Checks or retrieves confidence level for confidence intervals.
- `check_hyp_min_sample(n: int, p: int = None)`: Checks minimum sample size for hypothesis testing.
- `check_hyp_min_samples(p1: float, p2: float, n1: int, n2: int, overall=False)`: Checks minimum sample sizes for hypothesis testing.

### Submodule: `hypothesis_validator`

This submodule validates hypotheses.

- `check_residuals_centered(residuals: list, alpha=None)`: Checks if a list is centered.
- `check_coefficients_non_zero(list_coeffs: list, list_coeff_std: list, nb_obs: int, debug=False, alpha=None)`: Computes non-zero tests for each coefficient.
- `check_equal_var(*samples, alpha=COMMON_ALPHA_FOR_HYPH_TEST)`: Checks if samples have equal variance.
- `check_zero_to_one_constraint(*args)`: Checks if values are between zero and one.
- `check_or_get_alpha_for_hyph_test(alpha=None)`: Checks or retrieves alpha value for hypothesis testing.
- `check_or_get_cf_for_conf_inte(confidence=None)`: Checks or retrieves confidence level for confidence intervals.
- `check_hyp_min_sample(n: int, p: int = None)`: Checks minimum sample size for hypothesis testing.
- `check_hyp_min_samples(p1: float, p2: float, n1: int, n2: int, overall=False)`: Checks minimum sample sizes for hypothesis testing.

## Module: `conf_inte_md`

This module handles confidence interval computations.

### Submodule: `confidence_interval`

This submodule provides functions for computing confidence intervals.

- `IC_PROPORTION_ONE(sample_size: int, parameter: float, confidence: float = None, method: str = None)`: Computes confidence interval for one proportion.
- `IC_MEAN_ONE(sample: list, t_etoile=None, confidence: float = None)`: Computes confidence interval for one mean.
- `IC_PROPORTION_TWO(p1, p2, N1, N2, confidence: float = None)`: Computes confidence interval for the difference in proportions.
- `IC_MEAN_TWO_PAIR(sample1, sample2, t_etoile=None, confidence: float = None)`: Computes confidence interval for the difference in means for paired data.
- `IC_MEAN_TWO_NOTPAIR(sample1, sample2, pool=False, confidence: float = None)`: Computes confidence interval for the difference in means for non-paired data.

### Submodule: `ci_estimators`

This submodule provides functions for estimating confidence intervals.

- `get_min_sample(moe: float, p=None, method=None, cf: float = None)`: Computes minimum sample size for a given margin of error.
- `CIE_ONE_PROPORTION(proportion, n, method, cf: float = None)`: Computes confidence interval for one proportion.
- `CIE_PROPORTION_TWO(p1, p2, n1, n2, cf: float = None)`: Computes confidence interval for the difference in proportions.
- `CIE_MEAN_ONE(n, mean_dist, std_sample, t_etoile=None, cf: float = None)`: Computes confidence interval for one mean.
- `CIE_MEAN_TWO(N1, N2, diff_mean, std_sample_1, std_sample_2, t_etoile=None, pool=False, cf: float = None)`: Computes confidence interval for the difference in means.

## Module: `hyp_testi_md`

This module handles hypothesis testing functionalities.

### Submodule: `hp_estimators`

This submodule provides functions for hypothesis testing estimations.

- `HPE_FROM_P_VALUE(tail: str = None, p_value=None, t_stat=None, p_hat=None, p0=None, std_stat_eval=None, alpha=None, test=z_test, ddl=0, onetail=False)`: Hypothesis testing from p-value.
- `HPE_PROPORTION_ONE(alpha, p0, proportion, n, tail=Tails.right)`: Hypothesis testing for one proportion.
- `HPE_PROPORTION_TW0(alpha, p1, p2, n1, n2, tail=Tails.middle, evcpp=False)`: Hypothesis testing for the difference in proportions between two populations.
- `HPE_MEAN_ONE(alpha, p0, mean_dist, n, std_sample, tail=Tails.right)`: Hypothesis testing for one mean.
- `HPE_MEAN_TWO_PAIRED(alpha, mean_diff_sample, n, std_diff_sample, tail=Tails.middle)`: Hypothesis testing for the difference in means between two paired samples.
- `HPE_MEAN_TWO_NOTPAIRED(alpha, diff_mean, N1, N2, std_sample_1, std_sample_2, pool=False, tail=Tails.middle)`: Hypothesis testing for the difference in means between two independent samples.
- `HPE_MEAN_MANY(*samples, alpha=None)`: Hypothesis testing for multiple means.

### Submodule: `hypothesis_testing`

This submodule handles hypothesis testing.

- `HP_PROPORTION_ONE(sample_size: int, parameter: float, p0: float, alpha: float, symb=Tails.SUP_SYMB)`: Hypothesis testing for one proportion.
- `HP_MEAN_ONE(p0: float, alpha: float, sample: list, symb=Tails.SUP_SYMB)`: Hypothesis testing for one mean.
- `HP_PROPORTION_TWO(alpha, p1, p2, N1, N2, symb=Tails.NEQ_SYMB, evcpp=False)`: Hypothesis testing for the difference in proportions between two populations.
- `HP_MEAN_TWO_PAIR(alpha, sample1, sample2, symb=Tails.NEQ_SYMB)`: Hypothesis testing for the difference in means between two paired samples.
- `HP_MEAN_TWO_NOTPAIR(alpha, sample1, sample2, symb=Tails.NEQ_SYMB, pool=False)`: Hypothesis testing for the difference in means between two independent samples.

## Module: `utils_md`

This module includes various utility functions for preprocessing and estimation.

### Submodule: `preprocessing`

This submodule provides functions for preprocessing data.

- `clear_list(L: list) -> ndarray`: Removes NaN values from a list.
- `clear_list_pair(L1, L2) -> Tuple[ndarray, ndarray]`: Removes NaN values from two lists.
- `clear_mat_vec(A, y) -> Tuple[ndarray, ndarray]`: Removes NaN values from a matrix and a corresponding vector.

### Submodule: `estimate_std`

This submodule estimates standard deviation.

- `estimate_std(sample)`: Estimates standard deviation using (n-1) estimator.

### Submodule: `compute_ppf_and_p_value`

This submodule computes probabilities and percentiles.

- `get_z_value(cf)`: Computes z-value based on confidence level.
- `get_t_value(cf, ddl)`: Computes t-value based on confidence level and degrees of freedom.
- `get_f_value(cf, ddl)`: Computes F-value based on confidence level and degrees of freedom.
- `get_p_value_from_tail(prob, tail, debug=False)`: Computes p-value based on cumulative distribution function.
- `get_p_value_z_test(Z: float, tail: str, debug=False)`: Computes p-value based on normal distribution.
- `get_p_value_t_test(Z: float, ddl, tail: str, debug: bool = False)`: Computes p-value based on Student's t-distribution.
- `get_p_value_f_test(Z: float, dfn: int, dfd: int, debug: bool = False)`: Computes p-value based on Fisher's F-distribution.
- `get_p_value(Z: float, tail: str, test: str, ddl: int = None, debug=False)`: Computes p-value based on specified distribution.

### Submodule: `constants`

This submodule defines various constants used in hypothesis testing and confidence interval computations.

- Class `Confidence_data`: Defines confidence data parameters.
- Class `Hypothesis_data`: Defines hypothesis data parameters.
- ...

### Submodule: `refactoring`

This submodule provides utilities for refactoring code.

- `norm_tail(tail: str)`: Normalizes tail symbols.
- `get_tail_from_symb(symb: str)`: Converts symbol to tail.
- ...
