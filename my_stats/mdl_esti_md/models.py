import os.path
import sys
from my_stats.hyp_vali_md.constraints import (check_or_get_alpha_for_hyph_test,
                                              check_zero_to_one_constraint)
from my_stats.mdl_esti_md.model_estimator import (ME_Normal_dist)

print('mdl_esti_md.models: import start...')
sys.path.append(os.path.abspath("."))

# data manipulation
sum_loc = sum
# estimator

# hyp_validation

print('mdl_esti_md.models: ---import end---')


def model_normal_dist(sample, alpha=None):
    alpha = check_or_get_alpha_for_hyph_test(alpha)
    m, std_estimator, s, Testresults = ME_Normal_dist(sample, alpha=alpha)

    return m, s


if __name__ == "__main__":
    pass
else:
    print = lambda *args: ""
