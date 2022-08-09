import sys, os.path

sys.path.append(os.path.abspath("."))

from numpy import power, random, sin
# testing
import unittest
import statsmodels.api as sm_api
from scipy.stats import (f_oneway)
from pandas import read_csv

# utils
from my_stats.utils_md.estimate_std import (estimate_std)
from my_stats.utils_md.refactoring import (Tails)
# confidence_interval
from my_stats.conf_inte_md.confidence_interval import (IC_MEAN_ONE,
                                                       IC_MEAN_TWO_NOTPAIR,
                                                       IC_MEAN_TWO_PAIR,
                                                       IC_PROPORTION_ONE,
                                                       IC_PROPORTION_TWO)
from my_stats.conf_inte_md.ci_estimators import (
    CIE_MEAN_ONE,
    CIE_MEAN_TWO,
    CIE_ONE_PROPORTION,
    get_min_sample  #pip install statsmodels
)
# test_hyothesis
from my_stats.hyp_testi_md.hp_estimators import (
    HPE_FROM_P_VALUE, HPE_MEAN_MANY, HPE_MEAN_ONE, HPE_MEAN_TWO_NOTPAIRED,
    HPE_MEAN_TWO_PAIRED, HPE_PROPORTION_ONE, HPE_PROPORTION_TW0)
from my_stats.hyp_testi_md.hypothesis_testing import (HP_MEAN_ONE,
                                                      HP_MEAN_TWO_NOTPAIR,
                                                      HP_PROPORTION_ONE,
                                                      HP_PROPORTION_TWO)
# hyp_validation
from my_stats.hyp_vali_md.constraints import (check_hyp_min_sample,
                                              check_hyp_min_samples,
                                              check_zero_to_one_constraint)
# model_estimator
from my_stats.mdl_esti_md.hp_estimators_regression import (
    HPE_REGRESSION_FISHER_TEST)
from my_stats.mdl_esti_md.model_estimator import (ME_Normal_dist,
                                                  ME_Regression)
from my_stats.mdl_esti_md.models import (model_normal_dist)
