import os.path
import sys
from my_stats.hyp_vali_md.constraints import check_zero_to_one_constraint
from my_stats.utils_md.constants import COMMON_ALPHA_FOR_HYPH_TEST
from my_stats.mdl_esti_md.model_estimator import (ME_Normal_dist)
from numpy import random
print('mdl_esti_md.models: import start...')

sys.path.append(os.path.abspath("."))


# estimator

# utils
# hyp_validation

print('mdl_esti_md.models: ---import end---')


def model_normal_dist(sample, alpha=COMMON_ALPHA_FOR_HYPH_TEST):
    check_zero_to_one_constraint(alpha)
    m, std_estimator, s, Testresults = ME_Normal_dist(sample, alpha=alpha)

    return m, s


if __name__ == "__main__":
    pass
else:
    print = lambda *args: ""
