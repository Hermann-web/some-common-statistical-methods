import unittest

from numpy import random

from my_stats.common import sm_api
from my_stats.conf_inte_md import (CIE_MEAN_ONE, CIE_MEAN_TWO,
                                   CIE_ONE_PROPORTION, IC_MEAN_ONE,
                                   IC_MEAN_TWO_NOTPAIR, IC_MEAN_TWO_PAIR,
                                   IC_PROPORTION_ONE, IC_PROPORTION_TWO,
                                   get_min_sample)
from my_stats.utils_md import estimate_std


class Test_estimators(unittest.TestCase):

    def test_moe(self):
        cf = 0.95
        proportion = 0.43
        N = 232
        p, moe, interval = CIE_ONE_PROPORTION(cf=cf,
                                              proportion=proportion,
                                              n=N,
                                              method="classic")
        # print(interval)
        # print("vs >> 0.0637\n")
        assert abs(moe - 0.0637) <= 0.001
        # print("passed")
        # so he uses cf = z_quantile(0.95) = 1.96
        interval2 = sm_api.stats.proportion_confint(N * proportion, N)
        # print(interval2)
        assert abs(interval[0] - interval2[0]) <= 10**(-10)
        assert abs(interval[1] - interval2[1]) <= 10**(-10)

    def test_min_sample(self):
        cf = 0.95
        moe = 0.04  # 4%
        min_sample = get_min_sample(cf=cf, moe=moe)
        # print("min_sample = ", min_sample)
        # print(">>> vs 601\n")
        assert abs(min_sample - 601) <= 1
        min_sample = get_min_sample(cf=0.98, moe=0.03)
        # print("min_sample = ", min_sample)
        # print(">>> vs 1503\n")
        assert abs(min_sample - 1503) <= 1
        # print("passed")

    def test_mean(self):
        cf = 0.9
        n = 340
        mean_dist = 0.084
        std_sample = 0.76
        data = CIE_MEAN_ONE(cf=cf,
                            n=n,
                            mean_dist=mean_dist,
                            std_sample=std_sample,
                            t_etoile=1.967)
        p, marginOfError, interval = data
        assert abs(p - 0.084) < 0.01
        assert abs(marginOfError - 0.0814) < 0.001

        # example from sem3/"one_mean:Testing about..."/13:03
        cf = 0.9
        n = 25
        mean_dist = 82.48
        std_sample = 15.08
        data = CIE_MEAN_ONE(cf=cf,
                            n=n,
                            mean_dist=mean_dist,
                            std_sample=std_sample)
        p, marginOfError, interval = data
        print("data:", data)
        assert abs(p - 82.48) < 0.05
        assert abs(marginOfError - 5.15) < 0.05

        cf = 0.9
        n = 20
        mean_dist = 17.3
        std_sample = 28.49
        data = CIE_MEAN_ONE(cf=cf,
                            n=n,
                            mean_dist=mean_dist,
                            std_sample=std_sample,
                            t_etoile=2.093)
        p, marginOfError, interval = data
        print("data:", data)
        assert abs(p - 17.3) < 0.5
        assert abs(marginOfError - 13.33) < 0.05
        # Note that 0 is not in the CI. So a test [H0:mean_dist=0;H1:mean_dist!=0] will reject the null because H1 is verified for all points in the CI

        # Question 1

        # A simple random sample of 500 undergraduates at a large university self-administered a political knowledge test, where the maximum score is 100 and the minimum score is 0. The mean score was 62.5, and the standard deviation of the scores was 10. What is a 95% confidence interval for the overall undergraduate mean at the university?
        data = CIE_MEAN_ONE(cf=0.95, n=500, mean_dist=62.5, std_sample=10)
        print(":", data)

    def test_diff_mean(self):
        cf = None
        t_etoile = 1.98
        N1, std_sample_1 = 258, 6.24
        N2, std_sample_2 = 239, 6.43
        diff_mean = 23.57 - 22.83  # order doest not matter here
        pool = True  # equal_var
        data = CIE_MEAN_TWO(cf=cf,
                            N1=N1,
                            N2=N2,
                            diff_mean=diff_mean,
                            std_sample_1=std_sample_1,
                            std_sample_2=std_sample_2,
                            t_etoile=t_etoile,
                            pool=pool)
        p, marginOfError, _ = data
        # print(data)
        assert abs(p - 0.74) < 0.01
        assert abs(marginOfError - 1.125) < 0.001


class Tests(unittest.TestCase):

    def test_diff_proportion(self):

        cf = 0.95
        p1, N1 = 30 / 295, 295
        p2, N2 = 20 / 500, 500
        data = IC_PROPORTION_TWO(confidence=cf, p1=p1, p2=p2, N1=N1, N2=N2)
        print(
            f"\n->difference between two mean: {p1} on size {N1} and {p2} on size {N2} ..."
        )
        print(f"difference = {data.parameter} in interval {data.interval}")

    def test_confidence_interval(self):

        parameter = 0.53  # 53%
        sample_size = 526
        confidence = 0.9  # 90% confidence interval
        data = IC_PROPORTION_ONE(sample_size=sample_size,
                                 parameter=parameter,
                                 confidence=confidence)

        print(
            f"\n->parameter={parameter} sample_size={sample_size} confidence = {confidence}..."
        )
        print("interval: ", data.interval)

        # print(data)
        # print(">>>vs (0.4942, 0.5657)\n")
        assert abs(data.interval[0] - 0.4942) <= 10**(-2)
        assert abs(data.interval[1] - 0.5657) <= 10**(-2)
        # print("all passed")
        print("confidence interval:", data.interval)

    def test_estimate_population_mean(self):
        print("\n->estimate mean of gamma ...")
        confidence = 0.9
        shape = 2
        scale = 2
        sample = random.gamma(shape=shape, scale=shape, size=1000)
        moy = shape * scale
        data = IC_MEAN_ONE(confidence=confidence,
                           sample=sample,
                           t_etoile=2.064)
        print("estimation:", data.parameter, ">>vs>> moy:", moy)
        print("margin: ", data.marginOfError, ">>vs std", estimate_std(sample))

        interval = data.interval
        print(interval)
        interval2 = sm_api.stats.DescrStatsW(sample).zconfint_mean()
        print(interval2)
        assert abs(interval[0] - interval2[0]) <= 0.5  # large mais normal
        assert abs(interval[1] - interval2[1]) <= 0.5  # large mais normal

    def test_estimate_diff_mean_for_pair_data(self):
        print("\n->estimate mean for a paired data ...")
        confidence = 0.9
        shape = 2
        scale = 2
        diff = 2
        diff_bruit_std = 0.05 * diff
        Sample1 = random.gamma(shape=shape, scale=scale, size=1000)
        Sample2 = Sample1 + diff + diff_bruit_std * \
            random.randn(len(Sample1))  # sample1 + un_ajout + du_bruit_blanc
        data = IC_MEAN_TWO_PAIR(confidence=confidence,
                                sample1=Sample1,
                                sample2=Sample2)
        print("estimation of diff:", data.parameter, ">>vs>> diff:", diff)
        print("margin: ", data.marginOfError, ">>vs bruit_std", diff_bruit_std)

    def test_diff_mean(self):
        print("\n->estimate mean for a nonpaired so independant data ...")
        cf = 0.95
        shape = 2
        scale = 2
        diff = 2
        diff_bruit_std = 0.05 * diff
        Sample1 = random.gamma(shape=shape, scale=scale, size=1000)
        Sample2 = Sample1 + diff + diff_bruit_std * \
            random.randn(len(Sample1))  # sample1 + un_ajout + du_bruit_blanc
        data = IC_MEAN_TWO_NOTPAIR(confidence=cf,
                                   sample1=Sample1,
                                   sample2=Sample2)
        print("estimation of diff:", data.parameter, ">>vs>> diff:", diff)
        print("margin: ", data.marginOfError, ">>vs bruit_std", diff_bruit_std)
        print(
            "oh no!! ce calcul ne sépare pas le bruit. Logic. Le bruit rentre dans le std mais le calcul ne l'écarte pas"
        )


if __name__ == "__main__":
    unittest.main()
