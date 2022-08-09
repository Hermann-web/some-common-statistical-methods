if __name__ == "__main__":
    import sys, os.path
    sys.path.append(os.path.abspath("."))

from common_imports import *


class Tests_hyp_estimators(unittest.TestCase):

    def test_proportion_one(self):
        print("---> test_proportion_one")
        alpha = 0.05
        p0 = 0.52
        #Last year, 52% of parents believe that electronics and social media was the cause of their teenager lack of sleep
        # IS it more thid year?? => ">"
        #H0: p = 52%
        #H1: p > 0.52
        p = 0.56
        #This year, 56% of parents believe that electronics and social media was the cause of their teenager lack of sleep

        N = 1018
        _, _, _, Z, p_val, reject = HPE_PROPORTION_ONE(alpha, p0, p, N)
        #print(f"Z={Z} p_val={p_val} test_passed={reject}")
        assert abs(Z - 2.554) <= 0.005
        assert abs(p_val - 0.0053) <= 0.0005
        assert reject == True

        _, p_val_sm = sm_api.stats.proportions_ztest(p * N,
                                                     N,
                                                     p0,
                                                     alternative='larger',
                                                     prop_var=0.52)
        #print("sm:",Z, p_val, p_val_sm)
        assert abs(p_val_sm - p_val) <= 0.00001

    def test_proportion_two(self):
        print("---> test_proportion_two")
        alpha = 0.05
        p1, N1 = 91 / 247, 247  #black children => 91
        p2, N2 = 120 / 308, 308  #hispanic ones => 120
        #print("p1-p2: ",p1-p2)
        _, _, _, Z, p_val, reject_null = HPE_PROPORTION_TW0(
            alpha, p1, p2, N1, N2)
        print(f"Z={Z} p_val={p_val} test_passed={reject_null}")
        z_ = -0.02118 / 0.0414619  # -0.482369
        print("Z = ", Z, "z_ = ", z_)
        assert abs(abs(Z) - abs(z_)) <= 0.05
        assert abs(p_val - 0.6093128) <= 0.005
        assert reject_null == False

    def test_mean_one(self):
        print("---> test_mean_one")
        alpha = 0.05
        p0 = 80  #null value#
        #a guess from something
        #H0: p = 80
        #H1: p > 80
        mean_sample = 82.48  #from a sample #best_estimate#
        std_sample = 15.06  #std error estimated

        N = 25
        _, _, _, Z, p_val, reject_null = HPE_MEAN_ONE(alpha,
                                                      p0,
                                                      mean_sample,
                                                      N,
                                                      std_sample,
                                                      tail=Tails.right)
        #print(f"Z={Z} p_val={p_val} test_passed={reject_null}")
        assert abs(Z - 0.82) <= 0.05
        assert abs(p_val - 0.21) <= 0.05
        assert reject_null == False  #not enough evidence to reject the null

    def test_mean_two_paired(self):
        print("---> test_mean_two_paired")
        alpha = 0.05
        mean_diff_sample = 17.30  #from a sample #best_estimate#
        std_diff_sample = 28.49  #std error estimated

        N = 20
        _, _, _, Z, p_val, reject_null = HPE_MEAN_TWO_PAIRED(
            alpha, mean_diff_sample, N, std_diff_sample)
        #print(f"Z={Z} p_val={p_val} test_passed={reject_null}")
        assert abs(Z - 2.72) <= 0.05
        assert abs(p_val - 0.014) <= 0.005
        assert reject_null == True  #not enough evidence to reject the null #A confidence interval at (cf=O.95 show an interval=17.3 +- 13.33 wictch is totally in H1. So Ho is rejected)

    def test_mean_two_nonpaired(self):
        print("---> test_mean_two_nonpaired")
        alpha = 0.05
        mean_diff_sample = 23.57 - 22.83  #from a sample #best_estimate#

        N1, N2 = 257, 238
        std_sample_1, std_sample_2 = 6.24, 6.43
        _, _, _, Z, p_val, reject_null = HPE_MEAN_TWO_NOTPAIRED(
            alpha,
            mean_diff_sample,
            N1,
            N2,
            std_sample_1,
            std_sample_2,
            pool=False,
            tail=Tails.middle)

        #print(f"Z={Z} p_val={p_val} test_passed={reject_null}")
        assert abs(Z - 1.3) <= 0.5
        assert abs(
            p_val - 0.1942
        ) <= 0.05  #calcul assez large quand même pour être très exact (1942 vs 1956)
        assert reject_null == False

    def test_hp_from_p_value(self):
        print("---> test_hp_from_p_value")
        HPE_FROM_P_VALUE(tail=Tails.middle, p_value=22, test="z_test")
        HPE_FROM_P_VALUE(tail=Tails.middle, p_value=22, test="t_test", ddl=3)
        data = HPE_FROM_P_VALUE(tail=Tails.middle, t_stat=33, test="z_test")
        self.assertTrue(
            data.reject_null
        )  #the t_stat is too large => away from 0 => the test should pass
        data = HPE_FROM_P_VALUE(tail=Tails.middle,
                                p_hat=20,
                                p0=0,
                                std_stat_eval=50,
                                test="z_test",
                                ddl=0)
        print(data)

    def test_anova(self):
        print("\n->test_anova_on_normal ...")
        z = random.normal(2, 3, 25)
        x, y = z[:len(z) // 2], z[len(z) // 2:]
        _, _, _, F, p_value, reject_null = HPE_MEAN_MANY(x, y)
        stat_, p_val_ = f_oneway(x, y)
        print("res: ", stat_, p_val_, "mine", F, p_value)
        assert abs(stat_ - F) < 0.001
        assert abs(p_val_ - p_value) < 0.1


class Tests_estimators(unittest.TestCase):

    def test_proportion(self):
        '''
                    ### One Population Proportion

        #### Research Question 

        In previous years 52% of parents believed that electronics and social media was the cause of their teenager’s lack of sleep. Do more parents today believe that their teenager’s lack of sleep is caused due to electronics and social media?

        Population: Parents with a teenager (age 13-18)
        Parameter of Interest: p
        Null Hypothesis: p = 0.52
        Alternative Hypthosis: p > 0.52 (note that this is a one-sided test)

        1018 Parents

        56% believe that their teenager’s lack of sleep is caused due to electronics and social media.
        '''
        print("\n-->test_proportion...")
        n = 1018
        pnull = .52
        phat = .56
        data = HP_PROPORTION_ONE(sample_size=n,
                                 parameter=phat,
                                 p0=pnull,
                                 alpha=0.05)
        #print(data)
        Z_, p_v_ = sm_api.stats.proportions_ztest(phat * n,
                                                  n,
                                                  pnull,
                                                  alternative='larger',
                                                  prop_var=pnull)
        assert abs(abs(data.Z) - abs(Z_)) < 0.005
        assert abs(data.p_value - p_v_) < 0.005

        p_v_ = sm_api.stats.binom_test(phat * n,
                                       n,
                                       pnull,
                                       alternative='larger')
        #print("p_v_ = ",p_v_)
        assert abs(data.p_value - p_v_) < 0.005

        #ttest_ind compare les mean mais on peut donner deux listes (de 0 et 1, pour distinguer les deux categories => mean==proportion !)
        '''sample = binomial(1,phat,n)
        Z_, p_v_ = sm_api.stats.ztest(sample, value = pnull, alternative = "larger")
        print(f"Z_={Z_}, p_v_={p_v_}")
        print(data)
        assert abs(abs(data.Z) - abs(Z_))<0.0000005
        assert abs(data.p_value - p_v_)<0.005'''

    def test_proportion_two_population(self):
        '''### Difference in Population Proportions

        #### Research Question

        Is there a significant difference between the population proportions of parents of black children and parents of Hispanic children who report that their child has had some swimming lessons?

        Populations: All parents of black children age 6-18 and all parents of Hispanic children age 6-18
        Parameter of Interest: p1 - p2, where p1 = black and p2 = hispanic
        Null Hypothesis: p1 - p2 = 0
        Alternative Hypthosis: p1 - p2 ≠= 0

        91 out of 247 (36.8%) sampled parents of black children report that their child has had some swimming lessons.

        120 out of 308 (38.9%) sampled parents of Hispanic children report that their child has had some swimming lessons.
        '''
        print("\n-->test_proportion_two_popolution...")
        # This example implements the analysis from the "Difference in Two Proportions" lecture videos
        n1, n2 = 247, 308  # sample sizes
        y1, y2 = 91, 120  # Number of parents reporting that their child had some swimming lessons
        p1, p2 = y1 / n1, y2 / n2  # Estimates of the population proportions
        data = HP_PROPORTION_TWO(alpha=0.05,
                                 p1=p1,
                                 p2=p2,
                                 N1=n1,
                                 N2=n2,
                                 evcpp=True)
        #print(data)
        assert abs(abs(data.Z) - abs(-0.5110)) <= 0.05
        assert abs(data.p_value - 0.6093128) <= 0.005

        #ttest_ind compare les mean mais on peut donner deux listes (de 0 et 1, pour distinguer les deux categories => mean==proportion !)
        #sound alike a bad idea
        '''Sample1 = binomial(1,p1,n1)
        Sample2 = binomial(1,p2,n2)
        Z_, p_v_,_ = sm_api.stats.ttest_ind(Sample1, Sample2, alternative="two-sided")
        print(f"Z_={Z_}, p_v_={p_v_},_={_} ")
        print(data)
        assert abs(abs(data.Z) - abs(Z_))<0.0000005
        assert abs(data.p_value - p_v_)<0.005'''

    def test_estimate_population_mean(self):
        '''
        ### One Population Mean

        #### Research Question 

        Is the average cartwheel distance (in inches) for adults more than 80 inches?

        Population: All adults
        Parameter of Interest: μ
        , population mean cartwheel distance. Null Hypothesis: μ = 80 Alternative Hypthosis: μ > 80

        25 Adults

        μ=82.46

        σ=15.06'''
        print('\n-->test_estimate_population_mean...')
        df = read_csv("data/Cartwheeldata.csv")
        pnull = 80

        Z_, p_v_ = sm_api.stats.ztest(df["CWDistance"],
                                      value=pnull,
                                      alternative="larger")
        data = HP_MEAN_ONE(p0=pnull, alpha=0.05, sample=df["CWDistance"])
        #print(data)
        #print(f"Z_={Z_}, p_v_={p_v_}")
        assert abs(abs(data.Z) - abs(Z_)) < 0.0000005
        assert abs(data.p_value - p_v_) < 0.005

    def test_estimate_diff_mean_for_nonpair_data(self):
        '''
        Difference in Population Means
        Research Question

        Considering adults in the NHANES data, do males have a significantly higher mean Body Mass Index than females?

        Population: Adults in the NHANES data.
        Parameter of Interest: μ1−μ2
        , Body Mass Index.
        Null Hypothesis: μ1=μ2
        Alternative Hypthosis: μ1≠μ2

        2976 Females μ1=29.94

        σ1=7.75

        2759 Male Adults
        μ2=28.78

        σ2=6.25

        μ1−μ2=1.16
        '''
        print("\n->estimate_diff_mean_for_nonpair_data...")
        url = "data/nhanes_2015_2016.csv"
        da = read_csv(url)
        females = da[da["RIAGENDR"] == 2]
        male = da[da["RIAGENDR"] == 1]
        Sample1 = females["BMXBMI"].dropna()
        Sample2 = male["BMXBMI"].dropna()

        data = HP_MEAN_TWO_NOTPAIR(alpha=0.05,
                                   sample1=Sample1,
                                   sample2=Sample2,
                                   pool=True)
        #print(data)

        # use statsmodels.api.stats.ztest
        Z_, p_v_ = sm_api.stats.ztest(
            Sample1, Sample2, usevar="pooled",
            alternative="two-sided")  #only "pooled" is supported
        #print(f"Z_={Z_}, p_v_={p_v_}")
        assert abs(abs(data.Z) - abs(Z_)) < 0.0000005
        assert abs(data.p_value - p_v_) < 0.005

        # use scipy.stats.ttest_ind
        from scipy import stats
        Z_, p_v_ = stats.ttest_ind(
            Sample1, Sample2,
            equal_var=True)  # two-sided #equal_var=True => pooled
        #print(f"Z_={Z_}, p_v_={p_v_},_={_} ")
        #print(data)
        assert abs(abs(data.Z) - abs(Z_)) < 0.0000005
        assert abs(data.p_value - p_v_) < 0.005

        # use statsmodels.api.stats.ttest_ind
        Z_, p_v_, _ = sm_api.stats.ttest_ind(Sample1,
                                             Sample2,
                                             usevar="pooled",
                                             alternative="two-sided")
        #print(f"Z_={Z_}, p_v_={p_v_},_={_} ")
        #print(data)
        assert abs(abs(data.Z) - abs(Z_)) < 0.0000005
        assert abs(data.p_value - p_v_) < 0.005

        #using CompareMeans
        bmi1 = sm_api.stats.DescrStatsW(Sample1)
        bmi2 = sm_api.stats.DescrStatsW(Sample2)
        Z_, p_v_ = sm_api.stats.CompareMeans(bmi1, bmi2).ztest_ind(
            usevar='pooled', alternative="two-sided")
        #print(f"Z_={Z_}, p_v_={p_v_}")
        #print(data)
        assert abs(abs(data.Z) - abs(Z_)) < 0.0000005
        assert abs(data.p_value - p_v_) < 0.005

        # use statsmodels.api.stats.ttest_ind with equal_var==False
        Z_unequal, p_v_unequal, _ = sm_api.stats.ttest_ind(
            Sample1, Sample2, usevar="unequal", alternative="two-sided")
        data_unequal = HP_MEAN_TWO_NOTPAIR(alpha=0.05,
                                           sample1=Sample1,
                                           sample2=Sample2,
                                           pool=False)
        assert abs(abs(data_unequal.Z) - abs(Z_unequal)) < 0.0000005
        assert abs(data_unequal.p_value - p_v_unequal) < 0.005

        #using CompareMeans with equal_var==False
        bmi1 = sm_api.stats.DescrStatsW(Sample1)
        bmi2 = sm_api.stats.DescrStatsW(Sample2)
        Z_, p_v_ = sm_api.stats.CompareMeans(bmi1, bmi2).ztest_ind(
            usevar='unequal', alternative="two-sided")
        print(f"Z_={Z_}, p_v_={p_v_}")
        print(data_unequal)
        assert abs(abs(data_unequal.Z) - abs(Z_)) < 0.0000005
        assert abs(data_unequal.p_value - p_v_) < 0.005


if __name__ == "__main__":
    unittest.main()
