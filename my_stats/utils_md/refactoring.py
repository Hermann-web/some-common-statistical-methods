from dataclasses import field, dataclass


@dataclass()
class Confidence_data:
    parameter: float
    sample_size: int  #int or tuple
    confidence: int  #coverage_probability P(parameter in intervale_confaiance) = confidence

    marginOfError: float  # ==interval[1] - parameter == parameter - interval[0] == "few" * "the standard error of this estimate"
    interval: tuple  #confidence interval = (lower confidence bound (LCB) , upper confidence bound (UCB))


@dataclass()
class Hypothesis_data:
    parameter: float = field(
        repr=False
    )  # value to test if "bigger than p0" or "different then p0" or "less than p0"
    pnull: float = field(repr=False)  # prior value to check against
    std_stat_eval: float  # std_stat_eval to use to approximate that [ (parameter - p0)/std_stat_eval for sample in samples] follow certain distribution (Z-normal, T-student, F-fisher)
    tail: str = field(repr=False)  #right, left,middle
    sample_size: int = field(repr=False)  #int or tuple
    alpha: int = field(
        repr=False
    )  #level of significance #impact only the value of reject_null
    Z: float  # Z = (parameter-pnull)/sdt_stat_eval is the convertion of "parameter" nito the concerned distribution (Z, T, F, ...)
    p_value: float  # if tail=right, P(dist>Z); if tail=left, P(dist<Z); if tail=middle, P(dist>|Z|)+P(dist<-|Z|);
    reject_null: bool  #if H0 is rejected


@dataclass()
class RegressionFisherTestData:
    # unexplained
    DFE: float = field(default=None)
    SSE: float = field(repr=False, default=None)  #(y_hat-y)**2
    MSE: float = field(default=None)
    # explained
    DFR: float = field(
        default=None
    )  #number of predictor variables (regressors, not including intercept)
    SSR: float = field(repr=False, default=None)  #(y_hat-y_mean)**2
    MSR: float = field(default=None)  #SSR/(k-1)
    # total variance
    DFT: float = field(
        repr=False, default=None)  #n-1 where n is the number of observztions
    SST: float = field(repr=False, default=None)
    MST: float = field(repr=False, default=None)
    # R_carre
    R_carre: float = field(repr=False, default=None)  #1-SSR/SST
    R_carre_adj: float = field(repr=False, default=None)  #1-MSR/MST
    # F test
    F_stat: float = field(repr=False, default=None)  #F=MSR/MSE
    p_value: float = field(default=None)  #1-quantile(fisher, ddl=(k-1,n-k), F)
    reject_null: float = field(repr=False, default=None)  #If F is large


@dataclass()
class HypothesisValidationData:
    testPassed: bool
    obj: dict = field(default=None)


class Tails:
    INF_SYMB = "p<p0"
    SUP_SYMB = "p>p0"
    NEQ_SYMB = "p!=p0"

    right = "tail-right"
    left = "tail-left"
    middle = "tail-middle"

    def norm_tail(tail: str):

        dict_tail = {
            "left": Tails.left,
            "right": Tails.right,
            "middle": Tails.middle
        }

        if tail in dict_tail.values():
            return tail

        if tail in dict_tail:
            return dict_tail[tail]
        else:
            raise Exception(f"Bad tail. get {tail}")

    def get_tail_from_symb(symb: str):
        dict_symb = {
            Tails.INF_SYMB: "left",
            Tails.SUP_SYMB: Tails.right,
            Tails.NEQ_SYMB: Tails.middle
        }

        if symb in dict_symb:
            return dict_symb[symb]
        else:
            raise Exception("symbol is wrong")
