import unittest

from my_stats.hyp_vali_md import (check_hyp_min_sample, check_hyp_min_samples,
                                  check_zero_to_one_constraint)


class Tests_constraints(unittest.TestCase):

    def test_moe(self):
        check_hyp_min_sample(12)
        check_zero_to_one_constraint(0.4)
        check_hyp_min_samples(0.4, 0.5, 123, 134)

        try:
            check_hyp_min_sample(6)
            raise Exception("should go wrong")
        except:
            pass

        try:
            check_zero_to_one_constraint(1.3)
            raise Exception("should go wrong")
        except:
            pass

        try:
            check_hyp_min_samples(0.01, 0.02, 123, 134)
            raise Exception("should go wrong")
        except:
            pass


if __name__ == "__main__":

    unittest.main()
