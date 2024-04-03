# Table of Contents

* [statanalysis](#statanalysis)
* [statanalysis.mdl\_esti\_md.prediction\_metrics](#statanalysis.mdl_esti_md.prediction_metrics)
  * [compute\_skew](#statanalysis.mdl_esti_md.prediction_metrics.compute_skew)
  * [compute\_kurtosis](#statanalysis.mdl_esti_md.prediction_metrics.compute_kurtosis)
  * [compute\_aic\_bic](#statanalysis.mdl_esti_md.prediction_metrics.compute_aic_bic)
  * [PredictionMetrics](#statanalysis.mdl_esti_md.prediction_metrics.PredictionMetrics)
    * [compute\_log\_likelihood](#statanalysis.mdl_esti_md.prediction_metrics.PredictionMetrics.compute_log_likelihood)
* [statanalysis.mdl\_esti\_md](#statanalysis.mdl_esti_md)
* [statanalysis.mdl\_esti\_md.prediction\_results](#statanalysis.mdl_esti_md.prediction_results)
  * [HPE\_REGRESSION\_FISHER\_TEST](#statanalysis.mdl_esti_md.prediction_results.HPE_REGRESSION_FISHER_TEST)
  * [compute\_logit\_regression\_results](#statanalysis.mdl_esti_md.prediction_results.compute_logit_regression_results)
* [statanalysis.mdl\_esti\_md.log\_reg\_example](#statanalysis.mdl_esti_md.log_reg_example)
* [statanalysis.mdl\_esti\_md.hp\_estimators\_regression](#statanalysis.mdl_esti_md.hp_estimators_regression)
  * [ComputeRegression](#statanalysis.mdl_esti_md.hp_estimators_regression.ComputeRegression)
    * [fit](#statanalysis.mdl_esti_md.hp_estimators_regression.ComputeRegression.fit)
* [statanalysis.mdl\_esti\_md.model\_estimator](#statanalysis.mdl_esti_md.model_estimator)
  * [ME\_Normal\_dist](#statanalysis.mdl_esti_md.model_estimator.ME_Normal_dist)
  * [ME\_Regression](#statanalysis.mdl_esti_md.model_estimator.ME_Regression)
  * [ME\_multiple\_regression](#statanalysis.mdl_esti_md.model_estimator.ME_multiple_regression)
* [statanalysis.common](#statanalysis.common)
* [statanalysis.utils\_md.preprocessing](#statanalysis.utils_md.preprocessing)
  * [clear\_list](#statanalysis.utils_md.preprocessing.clear_list)
  * [clear\_list\_pair](#statanalysis.utils_md.preprocessing.clear_list_pair)
  * [clear\_mat\_vec](#statanalysis.utils_md.preprocessing.clear_mat_vec)
* [statanalysis.utils\_md](#statanalysis.utils_md)
* [statanalysis.utils\_md.estimate\_std](#statanalysis.utils_md.estimate_std)
  * [estimate\_std](#statanalysis.utils_md.estimate_std.estimate_std)
* [statanalysis.utils\_md.compute\_ppf\_and\_p\_value](#statanalysis.utils_md.compute_ppf_and_p_value)
  * [get\_p\_value\_from\_tail](#statanalysis.utils_md.compute_ppf_and_p_value.get_p_value_from_tail)
  * [get\_p\_value\_z\_test](#statanalysis.utils_md.compute_ppf_and_p_value.get_p_value_z_test)
  * [get\_p\_value\_t\_test](#statanalysis.utils_md.compute_ppf_and_p_value.get_p_value_t_test)
  * [get\_p\_value\_f\_test](#statanalysis.utils_md.compute_ppf_and_p_value.get_p_value_f_test)
  * [get\_p\_value](#statanalysis.utils_md.compute_ppf_and_p_value.get_p_value)
* [statanalysis.utils\_md.constants](#statanalysis.utils_md.constants)
* [statanalysis.utils\_md.refactoring](#statanalysis.utils_md.refactoring)
  * [Confidence\_data](#statanalysis.utils_md.refactoring.Confidence_data)
    * [sample\_size](#statanalysis.utils_md.refactoring.Confidence_data.sample_size)
  * [Hypothesis\_data](#statanalysis.utils_md.refactoring.Hypothesis_data)
    * [pnull](#statanalysis.utils_md.refactoring.Hypothesis_data.pnull)
    * [tail](#statanalysis.utils_md.refactoring.Hypothesis_data.tail)
    * [sample\_size](#statanalysis.utils_md.refactoring.Hypothesis_data.sample_size)
    * [reject\_null](#statanalysis.utils_md.refactoring.Hypothesis_data.reject_null)
  * [RegressionFisherTestData](#statanalysis.utils_md.refactoring.RegressionFisherTestData)
    * [MSR](#statanalysis.utils_md.refactoring.RegressionFisherTestData.MSR)
    * [R\_carre](#statanalysis.utils_md.refactoring.RegressionFisherTestData.R_carre)
    * [R\_carre\_adj](#statanalysis.utils_md.refactoring.RegressionFisherTestData.R_carre_adj)
    * [F\_stat](#statanalysis.utils_md.refactoring.RegressionFisherTestData.F_stat)
    * [reject\_null](#statanalysis.utils_md.refactoring.RegressionFisherTestData.reject_null)
* [statanalysis.hyp\_vali\_md](#statanalysis.hyp_vali_md)
* [statanalysis.hyp\_vali\_md.constraints](#statanalysis.hyp_vali_md.constraints)
  * [check\_sample\_normality](#statanalysis.hyp_vali_md.constraints.check_sample_normality)
  * [check\_equal\_var](#statanalysis.hyp_vali_md.constraints.check_equal_var)
* [statanalysis.hyp\_vali\_md.hypothesis\_validator](#statanalysis.hyp_vali_md.hypothesis_validator)
  * [check\_residuals\_centered](#statanalysis.hyp_vali_md.hypothesis_validator.check_residuals_centered)
  * [check\_coefficients\_non\_zero](#statanalysis.hyp_vali_md.hypothesis_validator.check_coefficients_non_zero)
  * [check\_equal\_mean](#statanalysis.hyp_vali_md.hypothesis_validator.check_equal_mean)
* [statanalysis.conf\_inte\_md](#statanalysis.conf_inte_md)
* [statanalysis.conf\_inte\_md.confidence\_interval](#statanalysis.conf_inte_md.confidence_interval)
  * [IC\_PROPORTION\_ONE](#statanalysis.conf_inte_md.confidence_interval.IC_PROPORTION_ONE)
  * [IC\_MEAN\_ONE](#statanalysis.conf_inte_md.confidence_interval.IC_MEAN_ONE)
  * [IC\_PROPORTION\_TWO](#statanalysis.conf_inte_md.confidence_interval.IC_PROPORTION_TWO)
  * [IC\_MEAN\_TWO\_PAIR](#statanalysis.conf_inte_md.confidence_interval.IC_MEAN_TWO_PAIR)
  * [IC\_MEAN\_TWO\_NOTPAIR](#statanalysis.conf_inte_md.confidence_interval.IC_MEAN_TWO_NOTPAIR)
* [statanalysis.conf\_inte\_md.ci\_estimators](#statanalysis.conf_inte_md.ci_estimators)
  * [get\_min\_sample](#statanalysis.conf_inte_md.ci_estimators.get_min_sample)
  * [CIE\_ONE\_PROPORTION](#statanalysis.conf_inte_md.ci_estimators.CIE_ONE_PROPORTION)
  * [CIE\_PROPORTION\_TWO](#statanalysis.conf_inte_md.ci_estimators.CIE_PROPORTION_TWO)
  * [CIE\_MEAN\_ONE](#statanalysis.conf_inte_md.ci_estimators.CIE_MEAN_ONE)
  * [CIE\_MEAN\_TWO](#statanalysis.conf_inte_md.ci_estimators.CIE_MEAN_TWO)
* [statanalysis.hyp\_testi\_md](#statanalysis.hyp_testi_md)
* [statanalysis.hyp\_testi\_md.hp\_estimators](#statanalysis.hyp_testi_md.hp_estimators)
  * [HPE\_FROM\_P\_VALUE](#statanalysis.hyp_testi_md.hp_estimators.HPE_FROM_P_VALUE)
  * [HPE\_PROPORTION\_ONE](#statanalysis.hyp_testi_md.hp_estimators.HPE_PROPORTION_ONE)
  * [HPE\_PROPORTION\_TW0](#statanalysis.hyp_testi_md.hp_estimators.HPE_PROPORTION_TW0)
  * [HPE\_MEAN\_ONE](#statanalysis.hyp_testi_md.hp_estimators.HPE_MEAN_ONE)
  * [HPE\_MEAN\_TWO\_PAIRED](#statanalysis.hyp_testi_md.hp_estimators.HPE_MEAN_TWO_PAIRED)
  * [HPE\_MEAN\_TWO\_NOTPAIRED](#statanalysis.hyp_testi_md.hp_estimators.HPE_MEAN_TWO_NOTPAIRED)
  * [HPE\_MEAN\_MANY](#statanalysis.hyp_testi_md.hp_estimators.HPE_MEAN_MANY)
* [statanalysis.hyp\_testi\_md.hypothesis\_testing](#statanalysis.hyp_testi_md.hypothesis_testing)
  * [HP\_PROPORTION\_ONE](#statanalysis.hyp_testi_md.hypothesis_testing.HP_PROPORTION_ONE)
  * [HP\_MEAN\_ONE](#statanalysis.hyp_testi_md.hypothesis_testing.HP_MEAN_ONE)
  * [HP\_PROPORTION\_TWO](#statanalysis.hyp_testi_md.hypothesis_testing.HP_PROPORTION_TWO)
  * [HP\_MEAN\_TWO\_PAIR](#statanalysis.hyp_testi_md.hypothesis_testing.HP_MEAN_TWO_PAIR)
  * [HP\_MEAN\_TWO\_NOTPAIR](#statanalysis.hyp_testi_md.hypothesis_testing.HP_MEAN_TWO_NOTPAIR)

<a id="statanalysis"></a>

# statanalysis

<a id="statanalysis.mdl_esti_md.prediction_metrics"></a>

# statanalysis.mdl\_esti\_md.prediction\_metrics

<a id="statanalysis.mdl_esti_md.prediction_metrics.compute_skew"></a>

#### compute\_skew

```python
def compute_skew(arr)
```

_summary_

**Arguments**:

- `y` __type__ - _description_
  
  Utils
  - [skewness and kurtosis - spcforexcel.com](https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics)
  - [skewness - thoughtco.com](https://www.thoughtco.com/what-is-skewness-in-statistics-3126242)
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.mdl_esti_md.prediction_metrics.compute_kurtosis"></a>

#### compute\_kurtosis

```python
def compute_kurtosis(arr, residuals=None)
```

_summary_

**Arguments**:

- `y` _list|array-like_ - _description_
  
  Utils
  - [kurtosis and skewness - spcforexcel.com](https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics)
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.mdl_esti_md.prediction_metrics.compute_aic_bic"></a>

#### compute\_aic\_bic

```python
def compute_aic_bic(dfr: int, n: int, llh: float, method: str = "basic")
```

_summary_

Utils
- It adds a penalty that increases the error when including additional terms. The lower the AIC, the better the model.
- [aic and bic in python - medium.com/analytics-vidhya](https://medium.com/analytics-vidhya/probabilistic-model-selection-with-aic-bic-in-python-f8471d6add32)

**Arguments**:

- `dfr` _int_ - nb_predictors(not including the intercept)
- `dfe` _int_ - nb of observations
- `llh` _float_ - log likelihood
  
  Question
  what about mixed models ?
  

**Returns**:

- `float` - aicself, y_true, y_pred

<a id="statanalysis.mdl_esti_md.prediction_metrics.PredictionMetrics"></a>

## PredictionMetrics Objects

```python
class PredictionMetrics()
```

<a id="statanalysis.mdl_esti_md.prediction_metrics.PredictionMetrics.compute_log_likelihood"></a>

#### compute\_log\_likelihood

```python
def compute_log_likelihood(std_eval: float = None,
                           debug=False,
                           min_tol: float = True)
```

_summary_

**Arguments**:

- `std_eval` _float, optional_ - (ignored if self.binary=True). Defaults to None.
- `debug` _bool, optional_ - _description_. Defaults to False.
- `min_tol` _float, optional_ - (ignored if self.binary=False). Defaults to None.
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.mdl_esti_md"></a>

# statanalysis.mdl\_esti\_md

<a id="statanalysis.mdl_esti_md.prediction_results"></a>

# statanalysis.mdl\_esti\_md.prediction\_results

<a id="statanalysis.mdl_esti_md.prediction_results.HPE_REGRESSION_FISHER_TEST"></a>

#### HPE\_REGRESSION\_FISHER\_TEST

```python
def HPE_REGRESSION_FISHER_TEST(y: list,
                               y_hat: list,
                               nb_param: int,
                               alpha: float = None)
```

check if mean is equal accross many samples

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
- [f_oneway - docs.scipy.org/doc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
- [anova and f test - blog.minitab.com](https://blog.minitab.com/fr/comprendre-lanalyse-de-la-variance-anova-et-le-test-f)
- [f-test-reg - facweb.cs.depaul.edu/sjost](http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm)

**Returns**:

- `data` - (RegressionFisherTestData)

<a id="statanalysis.mdl_esti_md.prediction_results.compute_logit_regression_results"></a>

#### compute\_logit\_regression\_results

```python
def compute_logit_regression_results(crd: RegressionResultData,
                                     debug: bool = False)
```

_summary_

**Arguments**:

- `crd` _RegressionResultData_ - _description_
- `debug` _bool, optional_ - _description_. Defaults to False.
  Info
  - [understand rs outputs - stats.stackexchange.com](https://stats.stackexchange.com/questions/86351/interpretation-of-rs-output-for-binomial-regression)
  - [pseudo-rcarre - stats.stackexchange.com](https://stats.stackexchange.com/questions/3559/which-pseudo-r2-measure-is-the-one-to-report-for-logistic-regression-cox-s)

**Returns**:

- `_type_` - _description_

<a id="statanalysis.mdl_esti_md.log_reg_example"></a>

# statanalysis.mdl\_esti\_md.log\_reg\_example

Author: Susan Li 
source: [LogisticRegressionImplementation.ipynb - github.com/aihubprojects](https://github.com/aihubprojects/Logistic-Regression-From-Scratch-Python/blob/master/LogisticRegressionImplementation.ipynb)

<a id="statanalysis.mdl_esti_md.hp_estimators_regression"></a>

# statanalysis.mdl\_esti\_md.hp\_estimators\_regression

todo
- refactor output (last lines)
- use "alternative" instead of "tail"
- use kwargs format while calling functions
- reorder fcts attributes
- Que signifie le R au carré négatif?: 
    - selon ma def, c'est  entre 0 et 1 à cause d'une somme mais c'est faux ?? [qastack.fr](https://qastack.fr/stats/183265/what-does-negative-r-squared-mean)

<a id="statanalysis.mdl_esti_md.hp_estimators_regression.ComputeRegression"></a>

## ComputeRegression Objects

```python
class ComputeRegression()
```

<a id="statanalysis.mdl_esti_md.hp_estimators_regression.ComputeRegression.fit"></a>

#### fit

```python
def fit(X, y, nb_iter: float = None, learning_rate: float = None)
```

_summary_

**Arguments**:

- `X` _2-dim array_ - list of columns (including slope) (n,nb_params)
- `y` _1-dim array_ - observations (n,)
- `alpha` __type_, optional_ - _description_. Defaults to None.
- `debug` _bool, optional_ - _description_. Defaults to False.
  

**Raises**:

- `Exception` - _description_
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.mdl_esti_md.model_estimator"></a>

# statanalysis.mdl\_esti\_md.model\_estimator

We know why t-student is useful
what about khi-2 ? we know 
fisher ? yes F

- add a fct to predict 
    - attention to extrapolation (unsern data) vs interpolation
- another for the curve showing the std
    - the interval should be narrower tinyer when X reacg the sample mean
- a good list of intel/reminder about the regression [here - sites.ualberta.ca - pdf](https://sites.ualberta.ca/~lkgray/uploads/7/3/6/2/7362679/slides_-_multiplelinearregressionaic.pdf)

<a id="statanalysis.mdl_esti_md.model_estimator.ME_Normal_dist"></a>

#### ME\_Normal\_dist

```python
def ME_Normal_dist(sample: list, alpha=None, debug=False)
```

estimate a normal distribution from a sample

visualisation:
- check if normal:
    - sns.distplot(data.X)
    - check if qq-plot is linear [en.wikipedia.org](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)
        ::from statsmodels.graphics.gofplots import qqplot
        ::from matplotlib import pyplot
        ::qqplot(sample, line='s')
        ::pyplot.show()

hypothesis
- X = m + N(0,s**2)

- check normal hypothesis: [machinelearningmastery](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)

lenght
- you may need data over 1000 samples to get

<a id="statanalysis.mdl_esti_md.model_estimator.ME_Regression"></a>

#### ME\_Regression

```python
def ME_Regression(x: list,
                  y: list,
                  degre: int,
                  logit=False,
                  fit_intercept=True,
                  debug=False,
                  alpha: float = 0.05,
                  nb_iter: int = 100000,
                  learning_rate: float = 0.1)
```

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
- [standard error of the intercept - stats.stackexchange](https://stats.stackexchange.com/questions/173271/what-exactly-is-the-standard-error-of-the-intercept-in-multiple-regression-analy)

<a id="statanalysis.mdl_esti_md.model_estimator.ME_multiple_regression"></a>

#### ME\_multiple\_regression

```python
def ME_multiple_regression(X: list,
                           y: list,
                           logit=False,
                           fit_intercept=True,
                           debug=False,
                           alpha: float = 0.05,
                           nb_iter: int = 100000,
                           learning_rate: float = 0.1)
```

_summary_

**Arguments**:

- `X` _list_ - _description_
- `y` _list_ - _description_
- `debug` _bool, optional_ - _description_. Defaults to False.
- `alpha` __type_, optional_ - _description_. Defaults to COMMON_ALPHA_FOR_HYPH_TEST.
  
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
  

**Raises**:

- `Exception` - _description_
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.common"></a>

# statanalysis.common

<a id="statanalysis.utils_md.preprocessing"></a>

# statanalysis.utils\_md.preprocessing

<a id="statanalysis.utils_md.preprocessing.clear_list"></a>

#### clear\_list

```python
def clear_list(L: list) -> ndarray
```

remove nan from a list

**Arguments**:

- `L` _list_ - a 1-dim array (n,). Anyway, data will be flatten
  
  What about he handle missing values properly !!
  - weight shit
  - Anyway, it would be good to know how missing values removal the distribution of L
  

**Returns**:

  1-dim array: array of shape (n,)
  
  Examples
  --------
  >>> A = np.array([
  [1,3],
  [4,3],
  [5,3],
  [7,np.nan]
  ])
  >>> y = np.array([6,np.nan,3,2])
  >>> A1 = clear_list(A)
  >>> y1 = clear_list(y)
  >>> print("A1: ",A1)
- `A1` - array([1, 3, 4, 3, 5, 3])
  >>> print("y: ",y1)
- `y` - array([6. 3. 2.])

<a id="statanalysis.utils_md.preprocessing.clear_list_pair"></a>

#### clear\_list\_pair

```python
def clear_list_pair(L1, L2) -> Tuple[ndarray, ndarray]
```

remove nan values (remove observation data containing nan value in L1 or L2) from 2 lists

**Arguments**:

- `L1` _list_ - a 1-dim array (n,). Anyway, data will be flatten
- `L2` _list_ - a 1-dim array (n,). Anyway, data will be flatten
  
  What about he handle missing values properly !!
  - weight shit
  - Anyway, it would be good to know how missing values removal the distribution of L
  

**Raises**:

  L1 and L2 have different size: lists must be of the same size
  

**Returns**:

  1-dim array: L1 of shape(n,)
  1-dim array: L2 of shape(n,)
  
  Examples
  --------
  >>> y1 = np.array([4, 8,np.nan,2])
  >>> y2 = np.array([6,np.nan,36,9])
  >>> y1,y2 = clear_list_pair(y1, y2)
  >>> print("y1: ",y1)
- `y1` - array([4, 2])
  >>> print("y2: ",y2)
- `y2` - array([6. 9.])

<a id="statanalysis.utils_md.preprocessing.clear_mat_vec"></a>

#### clear\_mat\_vec

```python
def clear_mat_vec(A, y) -> Tuple[ndarray, ndarray]
```

Remove nan values (remove observation data containing nan value in X or y) from a matric and a corresponding vector


Parameters
----------
A : 2-dimensional array (n,p)
y: 1-dimensional array (n,)

Others
----------
What about he handle missing values properly !!
    - weight shit
    - Anyway, it would be good to know how missing values removal the distribution of L

Raises
---------
    L1 and L2 have different size: lists must be of the same size

Returns
-----------
    1-dim array: L1 of shape(n,)
    1-dim array: L2 of shape(n,)

Examples
--------
>>> A = np.array([
        [1,3],
        [4,3],
        [5,3],
        [7,np.nan]
        ])
>>> y = np.array([6,np.nan,3,2])
>>> A1,y1 = clear_mat_vec(A,y)
>>> print("A1: ",A1)
A1:  [[1. 3.]
     [5. 3.]]
>>> print("y: ",y1)
y:  [6. 3.]

<a id="statanalysis.utils_md"></a>

# statanalysis.utils\_md

<a id="statanalysis.utils_md.estimate_std"></a>

# statanalysis.utils\_md.estimate\_std

<a id="statanalysis.utils_md.estimate_std.estimate_std"></a>

#### estimate\_std

```python
def estimate_std(sample)
```

Instead of std, he divide by (n-1) correspondng to the std estimator used in t-test

<a id="statanalysis.utils_md.compute_ppf_and_p_value"></a>

# statanalysis.utils\_md.compute\_ppf\_and\_p\_value

<a id="statanalysis.utils_md.compute_ppf_and_p_value.get_p_value_from_tail"></a>

#### get\_p\_value\_from\_tail

```python
def get_p_value_from_tail(prob, tail, debug=False)
```

get p value based on cdf and tail
If tail=Tails.middle, the distribution is assumed symmetric because we double F(Z)
if tail
    - right: return P(N > Z) = 1- F(Z) =  1 - prob
    - left: return P(N < Z) = F(Z) = prob
    - middle: return P(N < -|Z|) + P(N > |Z|) => return  2*P(N > |Z|)

<a id="statanalysis.utils_md.compute_ppf_and_p_value.get_p_value_z_test"></a>

#### get\_p\_value\_z\_test

```python
def get_p_value_z_test(Z: float, tail: str, debug=False)
```

get p value based on normal distribution N(0, 1)
if tail
    - right: return P(N > Z)
    - left: return P(N < Z)
    - middle: return P(N < -|Z|) + P(N > |Z|) => return  2*P(N > |Z|)

<a id="statanalysis.utils_md.compute_ppf_and_p_value.get_p_value_t_test"></a>

#### get\_p\_value\_t\_test

```python
def get_p_value_t_test(Z: float, ddl, tail: str, debug: bool = False)
```

get p value based on student distribution T(df=ddl) with ddl degres of freedom
if tail
    - right: return P(T > Z)
    - left: return P(T < Z)
    - middle: return P(T < -|Z|) + P(T > |Z|) => return  2*P(T > |Z|)

<a id="statanalysis.utils_md.compute_ppf_and_p_value.get_p_value_f_test"></a>

#### get\_p\_value\_f\_test

```python
def get_p_value_f_test(Z: float, dfn: int, dfd: int, debug: bool = False)
```

get p value based on fisher distribution T(dfn, dfd) with ddl degres of freedom

Utils
- [F-distribution - wiki](https://en.wikipedia.org/wiki/F-distribution)
- tail is right because Fisher is positive

**Arguments**:

- `Z` _float_ - _description_
- `dfn` _int_ - _description_
- `dfd` _int_ - _description_
- `debug` _bool, optional_ - _description_. Defaults to False.
  

**Raises**:

- `Exception` - _description_
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.utils_md.compute_ppf_and_p_value.get_p_value"></a>

#### get\_p\_value

```python
def get_p_value(Z: float, tail: str, test: str, ddl: int = None, debug=False)
```

get p value based on
    - (if test=="t_test") student distribution T(df=ddl) with ddl degres of freedom
    - (if test=="z_test") normal distribution N(0, 1)
    - (if test=="f_test") normal distribution F(ddl[0], ddl[1])

if tail
    - right: return P(T > Z)
    - left: return P(T < Z)
    - middle: return P(T < -|Z|) + P(T > |Z|) => return  2*P(T > |Z|)

<a id="statanalysis.utils_md.constants"></a>

# statanalysis.utils\_md.constants

<a id="statanalysis.utils_md.refactoring"></a>

# statanalysis.utils\_md.refactoring

<a id="statanalysis.utils_md.refactoring.Confidence_data"></a>

## Confidence\_data Objects

```python
@dataclass()
class Confidence_data()
```

<a id="statanalysis.utils_md.refactoring.Confidence_data.sample_size"></a>

#### sample\_size

int or tuple

<a id="statanalysis.utils_md.refactoring.Hypothesis_data"></a>

## Hypothesis\_data Objects

```python
@dataclass()
class Hypothesis_data()
```

<a id="statanalysis.utils_md.refactoring.Hypothesis_data.pnull"></a>

#### pnull

prior value to check against

<a id="statanalysis.utils_md.refactoring.Hypothesis_data.tail"></a>

#### tail

right, left,middle

<a id="statanalysis.utils_md.refactoring.Hypothesis_data.sample_size"></a>

#### sample\_size

int or tuple

<a id="statanalysis.utils_md.refactoring.Hypothesis_data.reject_null"></a>

#### reject\_null

if H0 is rejected

<a id="statanalysis.utils_md.refactoring.RegressionFisherTestData"></a>

## RegressionFisherTestData Objects

```python
@dataclass()
class RegressionFisherTestData()
```

<a id="statanalysis.utils_md.refactoring.RegressionFisherTestData.MSR"></a>

#### MSR

SSR/(k-1)

<a id="statanalysis.utils_md.refactoring.RegressionFisherTestData.R_carre"></a>

#### R\_carre

1-SSR/SST

<a id="statanalysis.utils_md.refactoring.RegressionFisherTestData.R_carre_adj"></a>

#### R\_carre\_adj

1-MSR/MST

<a id="statanalysis.utils_md.refactoring.RegressionFisherTestData.F_stat"></a>

#### F\_stat

F=MSR/MSE

<a id="statanalysis.utils_md.refactoring.RegressionFisherTestData.reject_null"></a>

#### reject\_null

If F is large

<a id="statanalysis.hyp_vali_md"></a>

# statanalysis.hyp\_vali\_md

<a id="statanalysis.hyp_vali_md.constraints"></a>

# statanalysis.hyp\_vali\_md.constraints

<a id="statanalysis.hyp_vali_md.constraints.check_sample_normality"></a>

#### check\_sample\_normality

```python
def check_sample_normality(residuals: list, debug=False, alpha=None)
```

check if residuals is like a normal distribution
- test_implemented


**Arguments**:

- `residuals` _list_ - list of float or array-like (will be flatten)
- `debug` _bool, optional_ - _description_. Defaults to False.
  

**Returns**:

- `bool` - if all tests passed

<a id="statanalysis.hyp_vali_md.constraints.check_equal_var"></a>

#### check\_equal\_var

```python
def check_equal_var(*samples, alpha=COMMON_ALPHA_FOR_HYPH_TEST)
```

_summary_

**Arguments**:

- `alpha` __type_, optional_ - _description_. Defaults to COMMON_ALPHA_FOR_HYPH_TEST.
  Utils
  - use levene test [plus robuste que fisher ou bartlett face à la non-normalité de la donnée](https://fr.wikipedia.org/wiki/Test_de_Bartlett)
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.hyp_vali_md.hypothesis_validator"></a>

# statanalysis.hyp\_vali\_md.hypothesis\_validator

les test de valisation (hypothese avant de lancer un autre test) qui dependant de test que j'ai écrits moi-même
Les mettre dans utils prut créer un import circulaire

<a id="statanalysis.hyp_vali_md.hypothesis_validator.check_residuals_centered"></a>

#### check\_residuals\_centered

```python
def check_residuals_centered(residuals: list, alpha=None)
```

check if a list is centered (if the mean ==0 nuder a significance od 0.05)

**Arguments**:

- `residuals` _list_ - list or array-like
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.hyp_vali_md.hypothesis_validator.check_coefficients_non_zero"></a>

#### check\_coefficients\_non\_zero

```python
def check_coefficients_non_zero(list_coeffs: list,
                                list_coeff_std: list,
                                nb_obs: int,
                                debug=False,
                                alpha=None)
```

compute non zero tests for each corfficien
- test
- for ech coefficient
- H0: coeff==0
- H1: coeff!=0
- if the test passed (H0 is rejected), the coefficient is away from 0, return = True

**Arguments**:

- `list_coeffs` _list_ - lists of values
- `list_coeff_std` _list_ - list of std; the two lists should have the same lenght
  

**Returns**:

  - HypothesisValidationData(pass_non_zero_test_bool,pass_non_zero_test)
  - testPassed (bool)
  - obj (list) list of boolean (For each value, True if H0 is reected)

<a id="statanalysis.hyp_vali_md.hypothesis_validator.check_equal_mean"></a>

#### check\_equal\_mean

```python
def check_equal_mean(*samples, alpha=None)
```

check if mean if the same accross samples

Hypothesis
H0: mean1 = mean2 = mean3 = ....
H1: one is different

Hypothesis
- The samples are independent.
- Each sample is from a normally distributed population.
- The population standard deviations of the groups are all equal. This property is known as homoscedasticity.

**Arguments**:

  - *samples (list): one or many lists
  
  Fisher test
  - The F Distribution is also called the Snedecor’s F, Fisher’s F or the Fisher–Snedecor distribution [1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html) [2](https://blog.minitab.com/fr/comprendre-lanalyse-de-la-variance-anova-et-le-test-f)
  

**Returns**:

- `stat` - (float) F
- `p_value` - (float)

<a id="statanalysis.conf_inte_md"></a>

# statanalysis.conf\_inte\_md

<a id="statanalysis.conf_inte_md.confidence_interval"></a>

# statanalysis.conf\_inte\_md.confidence\_interval

Some defs
- parameter: A quantifiable characteristic of a population
- confidence interval: range of reasonable values for the parameter

<a id="statanalysis.conf_inte_md.confidence_interval.IC_PROPORTION_ONE"></a>

#### IC\_PROPORTION\_ONE

```python
def IC_PROPORTION_ONE(sample_size: int,
                      parameter: float,
                      confidence: float = None,
                      method: str = None)
```

Confidence_interval(ONE PROPORTION):Confidence interval calculus after a statistic test
- input
- sample_size: int: sample size (more than 10 to use this method)
- parameter: float: the measurement on the sample
- confidence: float: confidence confidence (between O and 1). Greater the confidence, wider the interval
- method: str: either "classic" (default) or "conservative.

- Example:
- how many men in the entire population with a con ?
- a form filled by 300 people show that there is only 120 men => p = (120/300); N=300

- Hypothesis
- the sample is over 10 for each of the categories in place => we use the "Law of Large Numbers"
- the sample proportion comes from data that is considered a simple random sample

- Idea
- let P: the real proportion in the population
- let S: Size of each sample == nb of observations per sample
- For many samples, we calculate proportions per sample: ex: for N samples of size S => N proportions values
- (p - P) / ( p*(1-p)/S ) follow a normal distribution

- Descriptions:
- For a given polulation and a parameter P to find, If we repeated this study many times, each producing a new sample (of the same size {res.sample_size==S}) from witch a {res.confidence} confidence interval is computed, then {res.confidence} of the resulting confidence intervals would be excpected to contain the true value P
- If the entire interval verify a property, then it is reasonable say that the parameter verify that property

- Result
- with a {res.confidence} confidence, we estimate that the populztion proportion who are men is between {res.left_tail} and {res.right_tail}

<a id="statanalysis.conf_inte_md.confidence_interval.IC_MEAN_ONE"></a>

#### IC\_MEAN\_ONE

```python
def IC_MEAN_ONE(sample: list, t_etoile=None, confidence: float = None)
```

Estimate_population_mean(ONE MEAN): We need the spread (std): We will use an estimation

Data
    - confidence:..
    - sample: value...

Method
- Use t-distribution to calculate few

Hypothesis
- Samples follow a normal (or large enough to bypass this assumption) => means of these sample follow a t-dist

<a id="statanalysis.conf_inte_md.confidence_interval.IC_PROPORTION_TWO"></a>

#### IC\_PROPORTION\_TWO

```python
def IC_PROPORTION_TWO(p1, p2, N1, N2, confidence: float = None)
```

Difference_population_proportion(TWO PROPORTIONS): We have have estimate a parameter p on two populations (1 , 2).How to estimate p1-p2 ? `p1`-p2#

Method
    - create joint confidence interval

Construction
- Cmparison

Hypotheses
- two independant random samples
- large enough sample sizes : 10 per category (ex 10 yes, 10 no)

<a id="statanalysis.conf_inte_md.confidence_interval.IC_MEAN_TWO_PAIR"></a>

#### IC\_MEAN\_TWO\_PAIR

```python
def IC_MEAN_TWO_PAIR(sample1,
                     sample2,
                     t_etoile=None,
                     confidence: float = None)
```

Difference_population_means_for_paired_data(TWO MEANS FOR PAIRED DATA): We have have estimate a parameter p on two populations (1 , 2).How to estimate p1-p2 ? `p1`-p2#


What is paired data ?

    - measurements took on individuals (people, home, any object)
    - technicality:
        - When in a dataset (len = n) there is a row  df.a witch values only repeat twice (=> df.a.nunique = n/2)
        - we can do a plot(x=feature1, y=feature2)
    - examples
        - Each home need canibet quote from two suppliers => we want to know if there is an average difference in nb_quotes from between twese two suppliers
        - In a blind taste test to compare two new juice flavors, grape and apple, consumers were given a sample of each flavor and the results will be used to estimate the percentage of all such consumers who prefer the grape flavor to the apple flavor.
    - Construction
        - It is like,
            - checking if a feature magnitude change when going from a category to another, each pair split the two cztegories
            - Example_contexte
                - having a dataframe df, with 3 col [name, score, equipe, role]
                    - equipe: "1" or "2"
                    - role: df.role.nunique = 11 => len(df)==22
                - Now there is a battle: For a "same role" fight", which team is the best?
            - Example_question
                - if education level are generally equal -> mean difference is 0
                    - Is there a mean difference between the education level of twins
                    - if education levels are unequel -> mean difference is not 0
                - So, Look for 0 in the ranfe of reaonable values

We need the spread (std): We will use an estimation

Equivl
- IC_MEAN_ONE(confidence, sample1 - sample2)

Data
    - confidence:..
    - Sample1: list: values...
    - Sample2: list: (same len) values...


Method
- Use t-distribution to calculate few
- create joint confidence interval

Hypothesis
- a random sample of identical twin sets
- Samples follow a normal (or large enough to bypass this assumption: (ex 20 twins)) => means of these sample follow a t-dist

Notes
- With {cf} confidence, the population mean difference of the (second_team - first_team) attribute is estimated to be between {data.interval[0]} and {dat.interval[1]}
- if all values are above 0, cool there is a significativity

<a id="statanalysis.conf_inte_md.confidence_interval.IC_MEAN_TWO_NOTPAIR"></a>

#### IC\_MEAN\_TWO\_NOTPAIR

```python
def IC_MEAN_TWO_NOTPAIR(sample1,
                        sample2,
                        pool=False,
                        confidence: float = None)
```

Difference_population_means_for_nonpaired_data(TWO MEANS FOR PAIRED DATA): We have have estimate a parameter p on two populations (1 , 2).How to estimate p1-p2 ? `p1`-p2#

Construction
- It is like,
    - checking if a feature magnitude change when going from a category to another
    - Example_contexte
        - having a dataframe df, with 3 col [name, score, equipe, role]
            - equipe: "1" or "2"
            - role: df.role.nunique = 11 => len(df)==22
        - Now there is a battle: For a "same role" fight", which team is the best?
    - Example_question
        - if education level are generally equal -> mean difference is 0
            - Is there a mean difference between the education level based on gender
            - if education levels are unequel -> mean difference is not 0
        - So, Look for 0 in the ranfe of reaonable values

We need the spread (std): We will use an estimation

Args
    - confidence:..
    - Sample1: list: values...
    - Sample2: list: (same len) values...
    - pool: default False
        - True
            - if we assume that our populations variance are equal
            - we use a t-distribution of (N1+N2-1) ddl
        - False
            - if we assume that our populations variance are not equal
            - we use a t-distribution of min(N1, N2)-1 ddl

Method
- Use t-distribution to calculate few
- create joint confidence interval

Hypothesis
- a random sample
- Samples follow a normal (or large enough to bypass this assumption: 10 per category) => means of these sample follow a t-dist

Notes
- With {cf} confidence, the population mean difference of the (second_team - first_team) attribute is estimated to be between {data.interval[0]} and {dat.interval[1]}
- if all values are above 0, cool there is a significativity

<a id="statanalysis.conf_inte_md.ci_estimators"></a>

# statanalysis.conf\_inte\_md.ci\_estimators

todo
- refactor output (last lines)
- use "alternative" instead of "tail"
- use kwargs format while calling functions
- reorder fcts attributes

<a id="statanalysis.conf_inte_md.ci_estimators.get_min_sample"></a>

#### get\_min\_sample

```python
def get_min_sample(moe: float, p=None, method=None, cf: float = None)
```

Get_min_sample:get the minimum of sample_size to use for a
- input
    - cf: confidence (or coverage_probability): between 0 and 1
    - moe: margin of error
    - method (optional): "conservative" (default)
    - p: not used if method=="conservative"
Hyp
- better the population follow nornal dist. Or use large sample (>10)

<a id="statanalysis.conf_inte_md.ci_estimators.CIE_ONE_PROPORTION"></a>

#### CIE\_ONE\_PROPORTION

```python
def CIE_ONE_PROPORTION(proportion, n, method, cf: float = None)
```

Get_interval_simple: get a proportion of an attribute value (male gender, ) in a population based on a sample  (no sign pb)
- cf: confidence_level (or coverage_probability)
- proportion: measurement
- n: number of observations == len(sample)
- method: "classic" or "conservative"

Hyp
- better the population follow nornal dist. Or use large sample (>10)

<a id="statanalysis.conf_inte_md.ci_estimators.CIE_PROPORTION_TWO"></a>

#### CIE\_PROPORTION\_TWO

```python
def CIE_PROPORTION_TWO(p1, p2, n1, n2, cf: float = None)
```

Get_interval_diff: get the diff of mean between 2 population based on a sample from each population (p1-p2) `p1`-p2#
- cf: confidence_level (or coverage_probability)
- p1: mean of liste1
- p2: mean of liste2
- n1: len(liste1)
- n2: len(liste2)

Hyp
- better the populations follow normal dist. Or use large samples (>10)

<a id="statanalysis.conf_inte_md.ci_estimators.CIE_MEAN_ONE"></a>

#### CIE\_MEAN\_ONE

```python
def CIE_MEAN_ONE(n, mean_dist, std_sample, t_etoile=None, cf: float = None)
```

Get_interval_mean:get the mean of a population from a sample  (no sign pb)
- cf: confidence level (or coverage_probability)
- n: number of observations == len(sample)
- mean_dist: the mean measured on the sample = mean(sample)
- std_sample: std of the sample  ==std(sample)
- t_etoile: if set, cf is ignored.

Hyp
- better the population follow nornal dist. Or use large sample (>10)
    - Alternative to normality: Wilcoxon Signed Rank Test

Theo
- reade [here](https://en.wikipedia.org/wiki/Student's_t-distribution#How_Student's_distribution_arises_from_sampling)

<a id="statanalysis.conf_inte_md.ci_estimators.CIE_MEAN_TWO"></a>

#### CIE\_MEAN\_TWO

```python
def CIE_MEAN_TWO(N1,
                 N2,
                 diff_mean,
                 std_sample_1,
                 std_sample_2,
                 t_etoile=None,
                 pool=False,
                 cf: float = None)
```

Get_interval_diff_mean: get  the diff in mean of two populations(taking their samples) (sign(diff_mean) => no sign pb)
- cf: confidence level (or coverage_probability)
- N1: number of observations == len(sample1)
- N2: number of observations == len(sample2)
- mean_dist: the mean measured on the sample = mean(sample)
- std_sample_1: std of the sample  ==std(sample1)
- std_sample_2: std of the sample  ==std(sample2)
- t_etoile: if set, cf is ignored.
- pool: default False
    - True
        - if we assume that our populations variance are equal
        - we use a t-distribution of (N1+N2-1) ddl
    - False
        - if we assume that our populations variance are not equal
        - we use a t-distribution of min(N1, N2)-1 ddl

Hyp
- both the population follow normal dist. Or use large sample (>10)
- the populations are independant from each other
- use simple random samples
- for pool=True, variances are assume to be the same
    - to test that, you can
        - use levene test [plus robuste que fusher ou bartlett face à la non-normalité de la donnée](https://fr.wikipedia.org/wiki/Test_de_Bartlett)
            - H0: Variances are equals; H1: there are not

            ```python
            scipy.stats.levene(liste1,liste2, center='mean')
            solution = "no equality" if p-value<0.05 else "equality"
            ```

        - or check if IQR are the same
            - IQR = quantile(75%) - quantile(25%)

Eqvl
- scipy.stats.ttest_ind(liste1,liste2, equal_var = False | True)


Eqvl_pointWise estimation
- Assume diff_mean = 82
- Result: diff_mean in CI = [77.33, 87.63]
- If we test H0:p=80 vs H1:p>80, we would fail to reject the null because H1 is not valide here
- As sa matter of fact, there is some value in CI below 80 witch if not compatible with H1 => the test doest give enough evidence to reject H0

Theo
- read [here](https://en.wikipedia.org/wiki/Student's_t-distribution#How_Student's_distribution_arises_from_sampling)

<a id="statanalysis.hyp_testi_md"></a>

# statanalysis.hyp\_testi\_md

<a id="statanalysis.hyp_testi_md.hp_estimators"></a>

# statanalysis.hyp\_testi\_md.hp\_estimators

utils 
- Dans un test, H0 est l'hypothese pessimiste 
    - il faudra donc assez d'evidence (p<0.05) afin de la rejeter

<a id="statanalysis.hyp_testi_md.hp_estimators.HPE_FROM_P_VALUE"></a>

#### HPE\_FROM\_P\_VALUE

```python
def HPE_FROM_P_VALUE(tail: str = None,
                     p_value=None,
                     t_stat=None,
                     p_hat=None,
                     p0=None,
                     std_stat_eval=None,
                     alpha=None,
                     test="z_test",
                     ddl=0,
                     onetail=False)
```

_summary_

**Arguments**:

- `tail` _str, optional_ - "middle" or "left" or "right"
- `p_value` __type_, optional_ - _description_. Defaults to None.
- `t_stat` __type_, optional_ - _description_. Defaults to None.
- `p_hat` __type_, optional_ - _description_. Defaults to None.
- `p0` __type_, optional_ - _description_. Defaults to None.
- `std_stat_eval` __type_, optional_ - _description_. Defaults to None.
- `alpha` __type_, optional_ - _description_. Defaults to None.
- `test` _str, optional_ - _description_. Defaults to "z_test".
- `ddl` _int, optional_ - _description_. Defaults to 0.
- `onetail` _bool, optional_ - if tail="middle". return one_tail_cf_p_value instead of the 2tail_2cf_p_value Defaults to False.
  
  
  

**Returns**:

- `_type_` - _description_

<a id="statanalysis.hyp_testi_md.hp_estimators.HPE_PROPORTION_ONE"></a>

#### HPE\_PROPORTION\_ONE

```python
def HPE_PROPORTION_ONE(alpha, p0, proportion, n, tail=Tails.right)
```

check a proportion of an attribute value (male gender, ) in a population based on a sample  (no sign pb) using a Z-statistic
- alpha: p_value_max: significance level
- p0: proportion under the null
- proportion: measurement
- n: number of observations == len(sample)
- tail:
    - right: check if p>p0
    - left: check if p<p0
    - middle: ckeck id p==p0
Hyp
- simple random sample
- large sample (np>10)

Hypotheses
- H0: proportion = p0
- H1:
    - tail==right => proportion > p0
    - tail==left => proportion < p0
    - tail==middle => proportion != p0

Detail
- use a normal distribion (Z-statistic)

Result (ex:tail=right)
- if reject==True
    - There is sufficient evidence to conclude that the population proportion of {....} is greater than p0

<a id="statanalysis.hyp_testi_md.hp_estimators.HPE_PROPORTION_TW0"></a>

#### HPE\_PROPORTION\_TW0

```python
def HPE_PROPORTION_TW0(alpha, p1, p2, n1, n2, tail=Tails.middle, evcpp=False)
```

check the diff of proportion between 2 population based on a sample from each population (p1-p2) `p1`-p2# using a Z-statistic (always used for difference of estimates).
there is also fisher and chi-square
- alpha: level of significance
- p1: proportion of liste1
- p2: proportion of liste2
- n1: len(liste1)
- n2: len(liste2)
- evcpp: bool(defult=False) (True -> Estimate of the variance of the combined population proportion)

Hyp
- two independant samples
- two random samples
- large enough data

Hypotheses
- H0: proportion = p0
- H1: proportion !=p0

Detail
- use a normal distribion (Z-statistic)

<a id="statanalysis.hyp_testi_md.hp_estimators.HPE_MEAN_ONE"></a>

#### HPE\_MEAN\_ONE

```python
def HPE_MEAN_ONE(alpha, p0, mean_dist, n, std_sample, tail=Tails.right)
```

get the mean of a population from a sample  (no sign pb) using using a T-statistic (always T for mean!! unless youre comparing a sample vs a population of known std)
- alpha:
- n: number of observations == len(sample)
- mean_dist: the mean measured on the sample = mean(sample)
- std_sample: std of the sample  ==std(sample). You should use a real estimate (ffod=n-1)

Hyp
- simple random sample
- better the population follow nornal dist. Or use large sample (>10)
    - Alternative to normality: Wilcoxon Signed Rank Test

Theo
- read [here](https://en.wikipedia.org/wiki/Student's_t-distribution#How_Student's_distribution_arises_from_sampling)

<a id="statanalysis.hyp_testi_md.hp_estimators.HPE_MEAN_TWO_PAIRED"></a>

#### HPE\_MEAN\_TWO\_PAIRED

```python
def HPE_MEAN_TWO_PAIRED(alpha,
                        mean_diff_sample,
                        n,
                        std_diff_sample,
                        tail=Tails.middle)
```

get the difference of mean between two list paired (no sign pb) using a T-statistic (always T for mean!! unless youre comparing a sample vs a population of known std)
- alpha:
- mean_diff_sample: the mean measured on the sample = mean(sample)
- n: number of observations == len(sample) == n1 == n2
- std_diff_sample: std of the sample  ==std(sample). You should use a real estimate (ffod=n-1)
- tail: default=Tails.middle to test the equality (mean_diff=0). But we can also to mean_diff>0 (right) or mean_diff<0 (left)

Hyp
- simple random sample
- better when the diff of the samples (sample1 - sample2) follow nornal dist. Or use large sample (>10)
- std_diff_sample is a good data based estimated [use (n-1) instead of n]. example: np.std(sample1 - sample2, ddof=1) is better than ddof=0 (default)

Hypothesis
- H0: p1 - p2 = 0
- H1:
    - H1: p1 - p2 != 0 for(tail=middle)
    - H1: p1 - p2 > 0 for(tail=right)
    - H1: p1 - p2 < 0 for(tail=left)

<a id="statanalysis.hyp_testi_md.hp_estimators.HPE_MEAN_TWO_NOTPAIRED"></a>

#### HPE\_MEAN\_TWO\_NOTPAIRED

```python
def HPE_MEAN_TWO_NOTPAIRED(alpha,
                           diff_mean,
                           N1,
                           N2,
                           std_sample_1,
                           std_sample_2,
                           pool=False,
                           tail=Tails.middle)
```

check the diff in mean of two populations(taking their samples) (sign(diff_mean) => no sign pb)
- alpha:
- N1: number of observations == len(sample1)
- N2: number of observations == len(sample2)
- mean_dist: the mean measured on the sample = mean(sample)
- std_sample_1: std of the sample  ==std(sample1)
- std_sample_2: std of the sample  ==std(sample2)
- pool: default False
    - True
        - if we assume that our populations variance are equal
        - we use a t-distribution of (N1+N2-1) ddl
    - False
        - if we assume that our populations variance are not equal
        - we use a t-distribution of min(N1, N2)-1 ddl

Hyp
- both the population follow normal dist. Or use large sample (>10)
- the populations are independant from each other
- use simple random samples
- for pool=True, variances must be the same
    - to test that, you can
        - use levene test [plus robuste que fusher ou bartlett face à la non-normalité de la donnée](https://fr.wikipedia.org/wiki/Test_de_Bartlett)
            ::H0: Variances are equals; H1: there are not
            ::scipy.stats.levene(liste1,liste2, center='mean')
            ::solution = "no equality" if p-value<0.05 else "equality"
        - or check if IQR are the same
            - IQR = quantile(75%) - quantile(25%)

Theo
- read [here](https://en.wikipedia.org/wiki/Student's_t-distribution#How_Student's_distribution_arises_from_sampling)

<a id="statanalysis.hyp_testi_md.hp_estimators.HPE_MEAN_MANY"></a>

#### HPE\_MEAN\_MANY

```python
def HPE_MEAN_MANY(*samples, alpha=None)
```

check if mean is equal accross many samples

Hypothesis
H0: mean1 = mean2 = mean3 = ....
H1: one is different

Hypothesis
- each sample is
- simple random
- normal
- indepebdant from others
- same variance
- if added, the "same variance test" should use levene test but apparently, use levene test [plus robuste que fusher ou bartlett face à la non-normalité de la donnée](https://fr.wikipedia.org/wiki/Test_de_Bartlett)


Fisher test
- The F Distribution is also called the Snedecor’s F, Fisher’s F or the Fisher–Snedecor distribution [1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html) [2](https://blog.minitab.com/fr/comprendre-lanalyse-de-la-variance-anova-et-le-test-f)

**Returns**:

- `stat` - (float) F
- `p_value` - (float)

<a id="statanalysis.hyp_testi_md.hypothesis_testing"></a>

# statanalysis.hyp\_testi\_md.hypothesis\_testing

utils 
    - Dans un test, H0 est l'hypothese pessimiste 
        - il faudra donc assez d'evidence (p<0.05) afin de la rejeter
        - on a alors mis une borne max faible sur l'erreur de type 1 (rejeter H0 alors qu'il est vrai)

Some defs
    - parameter: A quantifiable characteristic of a population (baseline)
    - alpha: level of significance = type1_error = proba(reject_null;when null is True)

<a id="statanalysis.hyp_testi_md.hypothesis_testing.HP_PROPORTION_ONE"></a>

#### HP\_PROPORTION\_ONE

```python
def HP_PROPORTION_ONE(sample_size: int,
                      parameter: float,
                      p0: float,
                      alpha: float,
                      symb=Tails.SUP_SYMB)
```

ONE PROPORTION:alpha calculus after a statistic test
- input
- sample_size: int: sample size (more than 10 to use this method)
- parameter: float: the measurement on the sample
- alpha: float: alpha alpha (between O and 1). Greater the alpha, wider the interval
- method: str: either "classic" (default) or "conservative.

- Example:
- how many men in the entire population with a con ?
- a form filled by 300 people show that there is only 120 men => p = (120/300); N=300

- Hypothesis
- the sample is over 10 for each of the categories in place => we use the "Law of Large Numbers"
- the sample proportion comes from data that is considered a simple random sample

- Idea
- let P: the real proportion in the population
- let S: Size of each sample == nb of observations per sample
- For many samples, we calculate proportions per sample: ex: for N samples of size S => N proportions values
- (p - P) / ( p*(1-p)/S ) follow a normal distribution

- Descriptions:
- For a given polulation and a parameter P to find, If we repeated this study many times, each producing a new sample (of the same size {res.sample_size==S}) from witch a {res.alpha} alpha is computed, then {res.alpha} of the resulting alphas would be excpected to contain the true value P
- If the entire interval verify a property, then it is reasonable say that the parameter verify that property

- Result
- with a {res.alpha} alpha, we estimate that the populztion proportion who are men is between {res.left_tail} and {res.right_tail}

<a id="statanalysis.hyp_testi_md.hypothesis_testing.HP_MEAN_ONE"></a>

#### HP\_MEAN\_ONE

```python
def HP_MEAN_ONE(p0: float, alpha: float, sample: list, symb=Tails.SUP_SYMB)
```

ONE MEAN: We need the spread (std): We will use an estimation

Data
    - alpha:..
    - sample: value...

Method
- Use t-distribution to calculate few

Hypothesis
- Samples follow a normal (or large enough to bypass this assumption) => means of these sample follow a t-dist

<a id="statanalysis.hyp_testi_md.hypothesis_testing.HP_PROPORTION_TWO"></a>

#### HP\_PROPORTION\_TWO

```python
def HP_PROPORTION_TWO(alpha, p1, p2, N1, N2, symb=Tails.NEQ_SYMB, evcpp=False)
```

TWO PROPORTIONS: We have have estimate a parameter p on two populations (1 , 2).How to estimate p1-p2 ? `p1`-p2#

Method
    - create joint alpha
    - evcpp: bool(defult=False) (True -> Estimate of the variance of the combined population proportion)


Construction
- Cmparison

Hypotheses
- two independant random samples
- large enough sample sizes : 10 per category (ex 10 yes, 10 no)

<a id="statanalysis.hyp_testi_md.hypothesis_testing.HP_MEAN_TWO_PAIR"></a>

#### HP\_MEAN\_TWO\_PAIR

```python
def HP_MEAN_TWO_PAIR(alpha, sample1, sample2, symb=Tails.NEQ_SYMB)
```

TWO MEANS FOR PAIRED DATA: We have have estimate a parameter p on two populations (1 , 2).How to estimate p1-p2 ? `p1`-p2#


What is paired data:
    - measurements took on individuals (people, home, any object)
    - technicality:
        - When in a dataset (len = n) there is a row  df.a witch values only repeat twice (=> df.a.nunique = n/2)
        - we can do a plot(x=feature1, y=feature2)
    - examples
        - Each home need canibet quote from two suppliers => we want to know if there is an average difference in nb_quotes from between twese two suppliers
        - In a blind taste test to compare two new juice flavors, grape and apple, consumers were given a sample of each flavor and the results will be used to estimate the percentage of all such consumers who prefer the grape flavor to the apple flavor.
    - Construction
        - It is like,
            - checking if a feature magnitude change when going from a category to another, each pair split the two cztegories
            - Example_contexte
                - having a dataframe df, with 3 col [name, score, equipe, role]
                    - equipe: "1" or "2"
                    - role: df.role.nunique = 11 => len(df)==22
                - Now there is a battle: For a "same role" fight", which team is the best?
            - Example_question
                - if education level are generally equal -> mean difference is 0
                    - Is there a mean difference between the education level of twins
                    - if education levels are unequel -> mean difference is not 0
                - So, Look for 0 in the ranfe of reaonable values

We need the spread (std): We will use an estimation

Equivl
- estimate_population_mean(alpha, sample1 - sample2)

Data
    - alpha:..
    - Sample1: list: values...
    - Sample2: list: (same len) values...


Method
- Use t-distribution to calculate few
- create joint alpha

Hypothesis
- a random sample of identical twin sets
- Samples follow a normal (or large enough to bypass this assumption: (ex 20 twins)) => means of these sample follow a t-dist

- description
- With {alpha} alpha, the population mean difference of the (second_team - first_team) attribute is estimated to be between {data.interval[0]} and {dat.interval[1]}
- if all values are above 0, cool there is a significativity

<a id="statanalysis.hyp_testi_md.hypothesis_testing.HP_MEAN_TWO_NOTPAIR"></a>

#### HP\_MEAN\_TWO\_NOTPAIR

```python
def HP_MEAN_TWO_NOTPAIR(alpha,
                        sample1,
                        sample2,
                        symb=Tails.NEQ_SYMB,
                        pool=False)
```

TWO MEANS FOR PAIRED DATA: We have have estimate a parameter p on two populations (1 , 2).How to check p1-p2 != 0? `p1`-p2#

Construction
- It is like,
    - checking if a feature magnitude change when going from a category to another
    - Example_contexte
        - having a dataframe df, with 3 col [name, score, equipe, role]
            - equipe: "1" or "2"
            - role: df.role.nunique = 11 => len(df)==22
        - Now there is a battle: For a "same role" fight", which team is the best?
    - Example_question
        - if education level are generally equal -> mean difference is 0
            - Is there a mean difference between the education level based on gender
            - if education levels are unequel -> mean difference is not 0
        - So, Look for 0 in the ranfe of reaonable values

We need the spread (std): We will use an estimation

Data
    - alpha:..
    - Sample1: list: values...
    - Sample2: list: (same len) values...
    - pool: default False
        - True
            - if we assume that our populations variance are equal
            - we use a t-distribution of (N1+N2-1) ddl
        - False
            - if we assume that our populations variance are not equal
            - we use a t-distribution of min(N1, N2)-1 ddl

Method
- Use t-distribution to calculate few
- create joint alpha

Hypothesis
- a random sample
- Samples follow a normal (or large enough to bypass this assumption: 10 per category) => means of these sample follow a t-dist

- description
- With {alpha} alpha, the population mean difference of the (second_team - first_team) attribute is estimated to be between {data.interval[0]} and {dat.interval[1]}
- if all values are above 0, cool there is a significativity

