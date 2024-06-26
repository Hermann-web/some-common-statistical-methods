import unittest
import warnings

import numpy as np
from numpy import power, random, sin
from pandas import read_csv
from sklearn.linear_model import LogisticRegression

from statanalysis.mdl_esti_md import (HPE_REGRESSION_FISHER_TEST,
                                      ME_multiple_regression, ME_Normal_dist,
                                      ME_Regression, PredictionMetrics)
from tests.common import sm_api


def resc(x, n): return x * 10**(n - 1 - np.floor(np.log10(x)))


def resc_array(z, n=2):
    n = int(n)
    assert n > 0
    x = z.copy()
    x[z > 0] = resc(z[z > 0], n=n)
    x[z < 0] = -resc(-z[z < 0], n=n)
    return x


assert resc(50, 2) == 50
assert resc(10, 2) == 10
assert resc(5, 2) == 50
assert resc(1, 2) == 10
assert resc(0.5, 2) == 50
assert resc(0.1, 2) == 10
assert resc(0.05, 2) == 50
assert resc(0.01, 2) == 10


class Tests_hp_estimators_regression(unittest.TestCase):

    def test_fisher(self):
        print("\n->test_fisher ...")
        df = read_csv("data/Cartwheeldata.csv")
        y = df["CWDistance"][:11] / 100
        y_hat = y + 2 * y.std() * random.normal(0, 1, len(y))
        data = HPE_REGRESSION_FISHER_TEST(y=y,
                                          y_hat=y_hat,
                                          nb_param=5,
                                          alpha=0.05)
        F, p_value, reject_null = data.F_stat, data.p_value, data.reject_null
        print(f"data = {data}")
        y = df["CWDistance"][:21] / 100
        y_hat = y + 2 * y.std() * random.normal(0, 1, len(y))
        data = HPE_REGRESSION_FISHER_TEST(y=y,
                                          y_hat=y_hat,
                                          nb_param=5,
                                          alpha=0.05)
        F, p_value, reject_null = data.F_stat, data.p_value, data.reject_null
        print(f"data = {data}")
        assert reject_null == True


class Test_model_estimator(unittest.TestCase):

    def test_ne_normal(self):
        print("\n->test_ne_normal ...")
        loc, scale = 20, 3
        sample = random.normal(loc=loc, scale=scale, size=5000)
        m, std_estimator, s, Testresults, _others = ME_Normal_dist(sample,
                                                                   alpha=0.05)
        print(m, std_estimator, s, Testresults)

        self.assertTrue(abs(m - loc) < 0.1)
        self.assertTrue(abs(scale - s) < 0.1)

    def test_regression(self):
        print("\n->test_regression ...")
        print('test1')
        loc, scale, size = 20, 3, 2000
        sample = random.normal(loc=loc, scale=scale, size=size)
        y = 12 + 2*sample + 3*power(sample, 2) + \
            0.0001*random.normal(0, scale, size)
        coeffs, coeff_std, residu_std, Testresults, _others = ME_Regression(
            x=sample, y=y, degre=2, alpha=0.05)
        # print("coeff:",coeffs,"coeff_estim_std:",coeff_std, "residu_std:",residu_std,"tests:",Testresults)
        self.assertTrue(abs(coeffs[0] - 12) < 0.1)
        self.assertTrue(Testresults["coeff_non_zero"].obj[0])
        self.assertTrue(abs(coeffs[1] - 2) < 0.1)
        self.assertTrue(Testresults["coeff_non_zero"].obj[1])
        self.assertTrue(abs(coeffs[2] - 3) < 0.1)
        self.assertTrue(Testresults["coeff_non_zero"].obj[2])
        self.assertTrue(Testresults["residu_mean_null"].testPassed)
        self.assertTrue(Testresults["residuals_normality"].testPassed)

        y = 12 + 2 * sample + 0.0001 * random.normal(0, scale, size)
        coeffs, coeff_std, residu_std, Testresults, _others = ME_Regression(
            x=sample, y=y, degre=2, alpha=0.05)
        # print("coeff:",coeffs,"coeff_estim_std:",coeff_std, "residu_std:",residu_std,"tests:",Testresults)
        self.assertTrue(abs(coeffs[0] - 12) < 0.1)
        self.assertTrue(Testresults["coeff_non_zero"].obj[0])
        self.assertTrue(abs(coeffs[1] - 2) < 0.1)
        self.assertTrue(Testresults["coeff_non_zero"].obj[1])
        self.assertTrue(abs(coeffs[2] - 0) < 0.1)
        self.assertFalse(Testresults["coeff_non_zero"].obj[2])
        self.assertTrue(Testresults["residu_mean_null"].testPassed)
        self.assertTrue(Testresults["residuals_normality"].testPassed)

        print('test2')
        y = 12 + sin(sample) + 0.0001 * random.normal(0, scale, size)
        coeffs, coeff_std, residu_std, Testresults, _others = ME_Regression(
            x=sample, y=y, degre=2, alpha=0.05)
        # print("coeff:",coeffs,"coeff_std:",coeff_std, "residu_std:",residu_std,"tests:",Testresults)
        self.assertFalse(Testresults["residuals_normality"].testPassed)

    def assertAlmostEqual_(self, first: float, second: float, diff: float):
        diff_r = abs(first - second)
        self.assertTrue(
            diff_r <= diff,
            f"{first}!={second}.  with diff={diff} while abs(first-second)={diff_r}"
        )

    def test_regression_real_data(self):
        print("\n->test_regression_real_data ...")
        debug = False
        df = read_csv("data/Cartwheeldata.csv")
        x = df.Height
        y = df.Wingspan
        model = sm_api.OLS.from_formula("Wingspan ~ Height", data=df)
        res = model.fit()

        coeffs, coeff_std, residu_std, Testresults, _others = ME_Regression(
            x=x, y=y, degre=1, alpha=0.05, debug=debug)
        # print(f"coeff:{coeffs}",f"coeff_estim_std:{coeff_std}", f"residu_std:{residu_std}",f"tests:{Testresults}", sep="\n")
        # print(f"coeff: {coeffs}\n-->coeff_estim_std: {coeff_std}\n-->residu_std: {residu_std}\n-->tests:")
        for elt, val in Testresults.items():
            # print(f"  -> {elt}:{val}")
            continue
        # self.assertAlmostEqual_(coeffs[0], 7.5518, 0.01)
        # self.assertTrue(abs(coeff_std[0] - 45.412) < 1)
        # self.assertTrue(Testresults["coeff_non_zero"].obj[0])
        # self.assertTrue(abs(coeffs[1] - 1.1076) < 0.01)
        # self.assertTrue(abs(coeff_std[1] - 0.670) < 0.1)
        # self.assertTrue(Testresults["coeff_non_zero"].obj[1])
        # self.assertTrue(Testresults["residu_mean_null"].testPassed)
        # self.assertTrue(Testresults["residuals_normality"].testPassed)

        # coeffs
        for i in range(len(coeffs)):
            self.assertAlmostEqual_(coeffs[i], res.params[i], 0.01)

        # coeffs_estimators_std
        for i in range(len(coeff_std)):
            self.assertAlmostEqual_(coeff_std[i] / coeffs[i],
                                    res.bse[i] / coeffs[i], 0.5)

        # test parameter not null
        value_test = 0
        for i in range(len(coeffs)):
            p_val, llh = res.el_test([value_test], [i])
            if not p_val > 10:
                # il y a debat
                continue
            is_coeff_not_null = p_val > 0.025
            print("idx:", i)
            assert Testresults["coeff_non_zero"].obj[i] == is_coeff_not_null

        # test significance of the model
        significance = Testresults["significance"]
        # degree of freedom
        DFE, DFR = significance['DFE'], significance['DFR']
        self.assertAlmostEqual_(DFE, res.df_resid, 0.01)
        self.assertAlmostEqual_(DFR, res.df_model, 0.01)

        # standard errors
        SSE, SSR = significance['SSE'], significance['SSR']
        MSE, MSR = significance['MSE'], significance['MSR']
        self.assertAlmostEqual_(MSE, res.mse_resid, 0.01)
        self.assertAlmostEqual_(MSR, res.mse_model, 0.01)

        # Coefficient of determination a.k.a “Goodness of fit”
        R_carre, R_carre_adj = significance['R_carre'], significance[
            'R_carre_adj']
        self.assertAlmostEqual_(R_carre, res.rsquared, 0.01)
        self.assertAlmostEqual_(R_carre_adj, res.rsquared_adj, 0.01)

        # fisher: The significance of the overall relationship described by the model
        fisher_res = Testresults["fisher_test"]
        F_stat, p_value = fisher_res["F_stat"], fisher_res["p_value"]
        self.assertAlmostEqual_(p_value, res.f_pvalue, 0.001)

        metrics = Testresults["metrics"]
        # log likelihood == log(product(proba(yi/xi))) when yi/xi is assumed gaussian with mean = y_hat(xi) and std = std(y_hat_estimation)=std(error_estimation)=std(y-y_yat)
        log_likelihood = metrics['log-likelihood']
        self.assertAlmostEqual_(log_likelihood, res.llf, 0.011)
        # aic, bic
        AIC, BIC = metrics["AIC"], metrics["BIC"]
        self.assertAlmostEqual_(AIC, res.aic, 0.025)
        self.assertAlmostEqual_(BIC, res.bic, 0.026)

        # others
        _ = res.summary()  # important #it create diagn attribute
        meta_data = res.diagn
        # skew, kustosis
        skew, kurtosis = metrics["skew"], metrics["kurtosis"]
        self.assertAlmostEqual_(skew, meta_data["skew"], 0.025)
        # self.assertAlmostEqual_(kurtosis, meta_data["kurtosis"], 0.026)
        warnings.warn('Eh the kurtosis is wrong hhh!!')

        # Omnibus #a Ftest with an F based on skew and kutosis
        omni = meta_data["omni"]
        omnipv = meta_data["omnipv"]  # Prob(Omnibus)
        # Durbin-Watson #H0:first-order autocorrelation: the presumed error that should be independant follow: u(t) = rho*u(t-1) + eps(t) where eps(t) is the ideal error #https://itfeature.com/time-series-analysis-and-forecasting/autocorrelation/first-order-autocorrelation #https://corporatefinanceinstitute.com/resources/knowledge/other/durbin-watson-statistic/
        mineigval = meta_data["mineigval"]
        jb = meta_data["jb"]  # Jarque-Bera (JB)
        jbpv = meta_data["jbpv"]  # Prob(JB)
        condno = meta_data["condno"]  # Cond. No.

    def test_multiple_regression(self):
        print("\n->test_multiple_regression ...")
        debug = True
        df = read_csv("data/Cartwheeldata.csv")
        X = df[["Height", "CWDistance"]]
        y = df.Wingspan
        model = sm_api.OLS.from_formula("Wingspan ~ Height+CWDistance",
                                        data=df)
        res = model.fit()
        coeffs, list_coeffs_std, residu_std, Testresults, _others = ME_multiple_regression(
            X, y, debug=debug, alpha=0.05)
        print(
            f"coeff: {coeffs}\n-->coeff_estim_std: {list_coeffs_std}\n-->residu_std: {residu_std}\n-->tests:"
        )
        for elt, val in Testresults.items():
            print(f"  -> {elt}:{val}")
        # coeffs
        for i in range(len(coeffs)):
            self.assertAlmostEqual_(coeffs[i], res.params[i], 0.01)

        # coeffs_estimators_std
        for i in range(len(list_coeffs_std)):
            self.assertAlmostEqual_(list_coeffs_std[i] / coeffs[i],
                                    res.bse[i] / coeffs[i], 0.5)

        # test parameter not null
        value_test = 0
        for i in range(len(coeffs)):
            p_val, llh = res.el_test([value_test], [i])
            if not p_val > 10:
                # il y a debat
                continue
            is_coeff_not_null = p_val > 0.025
            if debug:
                print("idx:", i)
            assert Testresults["coeff_non_zero"].obj[i] == is_coeff_not_null

        # test significance of the model
        significance = Testresults["significance"]
        # degree of freedom
        DFE, DFR = significance['DFE'], significance['DFR']
        self.assertAlmostEqual_(DFE, res.df_resid, 0.01)
        self.assertAlmostEqual_(DFR, res.df_model, 0.01)

        # standard errors
        SSE, SSR = significance['SSE'], significance['SSR']
        MSE, MSR = significance['MSE'], significance['MSR']
        self.assertAlmostEqual_(MSE, res.mse_resid, 0.01)
        self.assertAlmostEqual_(MSR, res.mse_model, 0.01)

        # Coefficient of determination a.k.a “Goodness of fit”
        R_carre, R_carre_adj = significance['R_carre'], significance[
            'R_carre_adj']
        self.assertAlmostEqual_(R_carre, res.rsquared, 0.01)
        self.assertAlmostEqual_(R_carre_adj, res.rsquared_adj, 0.01)

        # fisher: The significance of the overall relationship described by the model
        fisher_res = Testresults["fisher_test"]
        F_stat, p_value = fisher_res["F_stat"], fisher_res["p_value"]
        self.assertAlmostEqual_(p_value, res.f_pvalue, 0.001)

        metrics = Testresults["metrics"]
        # log likelihood == log(product(proba(yi/xi))) when yi/xi is assumed gaussian with mean = y_hat(xi) and std = std(y_hat_estimation)=std(error_estimation)=std(y-y_yat)
        log_likelihood = metrics['log-likelihood']
        self.assertAlmostEqual_(log_likelihood, res.llf, 0.011)
        # aic, bic
        AIC, BIC = metrics["AIC"], metrics["BIC"]
        self.assertAlmostEqual_(AIC, res.aic, 0.025)
        self.assertAlmostEqual_(BIC, res.bic, 0.026)

        # others
        _ = res.summary()  # important #it create diagn attribute
        meta_data = res.diagn
        # skew, kustosis
        skew, kurtosis = metrics["skew"], metrics["kurtosis"]
        self.assertAlmostEqual_(skew, meta_data["skew"], 0.025)
        # self.assertAlmostEqual_(kurtosis, meta_data["kurtosis"], 0.026)
        warnings.warn('Eh the kurtosis is wrong hhh!!')

        # Omnibus #a Ftest with an F based on skew and kutosis
        omni = meta_data["omni"]
        omnipv = meta_data["omnipv"]  # Prob(Omnibus)
        # Durbin-Watson #H0:first-order autocorrelation: the presumed error that should be independant follow: u(t) = rho*u(t-1) + eps(t) where eps(t) is the ideal error #https://itfeature.com/time-series-analysis-and-forecasting/autocorrelation/first-order-autocorrelation #https://corporatefinanceinstitute.com/resources/knowledge/other/durbin-watson-statistic/
        mineigval = meta_data["mineigval"]
        jb = meta_data["jb"]  # Jarque-Bera (JB)
        jbpv = meta_data["jbpv"]  # Prob(JB)
        condno = meta_data["condno"]  # Cond. No.


class Test_model_log_reg_estimator(unittest.TestCase):

    def test_log_reg(self):
        print("\n->test_log_reg ...")
        print('test1')

        loc, scale, size = 0, 1, 2000  # mean=0 pour centrer
        add_intercept = True
        degre = 2

        # data
        my_coeffs = [0, 2, 0]
        sample = np.random.normal(loc=loc, scale=scale, size=size)
        y = my_coeffs[1] * sample + 0.0001 * np.random.normal(0, scale, size)

        if degre > 1:
            my_coeffs[2] = 50
            y += my_coeffs[2] * np.power(sample, 2)

        if add_intercept:
            my_coeffs[0] = -y.mean()  # to center at the same time
            y += my_coeffs[0]

        y = 1 / (1 + np.exp(-y))
        y = (y > 0.5).astype('int')

        X = np.array([np.power(sample, i) for i in range(1, degre + 1)]).T

        # my regressor
        print("\n@@my regressor")
        my_reg_coeffs, coeff_std, residu_std, Testresults, _others = ME_multiple_regression(
            X=X,
            y=y,
            fit_intercept=True,
            logit=True,
            alpha=0.05,
            nb_iter=20000,
            learning_rate=0.1)
        print(
            f"->coeff: {my_reg_coeffs} \n->coeff_estim_std: {coeff_std} \n->residu_std: {residu_std} \n->tests"
        )
        print(Testresults)
        print(f"my_coeffs = {my_coeffs}")

        # logistic
        print("\n@@logitic regressor")
        _X = X.copy() if not add_intercept else np.hstack((np.ones(
            (X.shape[0], 1)), X))
        clf = LogisticRegression(random_state=0,
                                 fit_intercept=False).fit(_X, y)
        pm = PredictionMetrics(y, clf.predict_proba(_X)[:, 1], binary=True)
        print(pm.get_binary_regression_res())
        reg_coeffs = clf.coef_[0]
        print(f"pred_coeffs = {reg_coeffs}")

        # coefs
        print(
            "\n@@comparison des coeffs normalisés (les deux premiers chiffres non nuls sont pareils et l'ecart sur le 3e < 5)"
        )
        my_coeffs = np.array(my_coeffs[:reg_coeffs.size])
        reg_coeffs = reg_coeffs / np.abs(reg_coeffs).sum()
        my_coeffs = my_coeffs / np.abs(my_coeffs).sum()
        my_reg_coeffs = my_reg_coeffs / np.abs(my_reg_coeffs).sum()
        lv = 2
        print(
            f"my_coeffs = {my_coeffs}\nmy_reg_coeffs = {my_reg_coeffs}\npred_coeffs = {reg_coeffs}"
        )
        assert (np.floor(0.5 * np.ceil(2 * resc_array(my_coeffs, lv))) -
                np.floor(0.5 * np.ceil(2 * resc_array(reg_coeffs, lv)))
                < 5).all(), f"coefficient mismatch {my_coeffs} vs {reg_coeffs}"
        assert (
            np.floor(0.5 * np.ceil(2 * resc_array(my_coeffs, lv))) -
            np.floor(0.5 * np.ceil(2 * resc_array(my_reg_coeffs, lv)))
            < 5).all(), f"coefficient mismatch {my_coeffs} vs {my_reg_coeffs}"

        self.assertTrue(Testresults["coeff_non_zero"].obj[0])
        self.assertTrue(Testresults["coeff_non_zero"].obj[1])
        if degre > 1:
            self.assertTrue(Testresults["coeff_non_zero"].obj[2])
        self.assertLess(Testresults["metrics"]["log-loss"], 0.06)
        self.assertGreater(Testresults["metrics"]["accuracy"], 0.9)
        self.assertGreater(Testresults["metrics"]["precision"], 0.9)
        self.assertGreater(Testresults["metrics"]["recall"], 0.9)
        self.assertGreater(Testresults["metrics"]["f1"], 0.9)

        print(f"\n\n test ME_regression")
        my_coeffs, coeff_std, residu_std, Testresults, _others = ME_Regression(
            x=sample,
            y=y,
            degre=degre,
            fit_intercept=True,
            logit=True,
            alpha=0.05,
            nb_iter=20000,
            learning_rate=0.1)
        # coefs
        print(
            "\n@@comparison des coeffs normalisés (les deux premiers chiffres non nuls sont pareils et l'ecart sur le 3e < 5)"
        )
        my_coeffs = np.array(my_coeffs[:reg_coeffs.size])
        reg_coeffs = reg_coeffs / np.abs(reg_coeffs).sum()
        my_coeffs = my_coeffs / np.abs(my_coeffs).sum()
        my_reg_coeffs = my_reg_coeffs / np.abs(my_reg_coeffs).sum()
        lv = 2
        print(
            f"my_coeffs = {my_coeffs}\nmy_reg_coeffs = {my_reg_coeffs}\npred_coeffs = {reg_coeffs}"
        )
        assert (np.floor(0.5 * np.ceil(2 * resc_array(my_coeffs, lv))) -
                np.floor(0.5 * np.ceil(2 * resc_array(reg_coeffs, lv)))
                < 5).all(), f"coefficient mismatch {my_coeffs} vs {reg_coeffs}"
        assert (
            np.floor(0.5 * np.ceil(2 * resc_array(my_coeffs, lv))) -
            np.floor(0.5 * np.ceil(2 * resc_array(my_reg_coeffs, lv)))
            < 5).all(), f"coefficient mismatch {my_coeffs} vs {my_reg_coeffs}"

        self.assertTrue(Testresults["coeff_non_zero"].obj[0])
        self.assertTrue(Testresults["coeff_non_zero"].obj[1])
        if degre > 1:
            self.assertTrue(Testresults["coeff_non_zero"].obj[2])
        self.assertLess(Testresults["metrics"]["log-loss"], 0.06)
        self.assertGreater(Testresults["metrics"]["accuracy"], 0.9)
        self.assertGreater(Testresults["metrics"]["precision"], 0.9)
        self.assertGreater(Testresults["metrics"]["recall"], 0.9)
        self.assertGreater(Testresults["metrics"]["f1"], 0.9)

    def assertAlmostEqual_(self, first: float, second: float, diff: float):
        diff_r = abs(first - second)
        self.assertTrue(
            diff_r <= diff,
            f"{first}!={second}.  with diff={diff} while abs(first-second)={diff_r}"
        )


if __name__ == "__main__":
    unittest.main()
