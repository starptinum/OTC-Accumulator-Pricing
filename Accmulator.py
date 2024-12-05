from EQModel import *
from util import time_decorator
from YieldCurve import *
import numpy as np


class Accumulator_MC:
    def __init__(self, eq_model: EQModel, yc: YieldCurve, t: int, ko_price: float, guarantee_period: int, monitoring_freq: int, gearing, accumulate_if_barrier=True, tol=1e-10, path=10000, antithetic=True, seed=0):
        """
        A class that models an Accumulator product with knock-out and guarantee period.

        Parameters:
        -----------
        eq_model : EQModel
            The equity model used to simulate price paths.
        yc : YieldCurve
            The yield curve for discounting cash flows or interest rate calculations.
        t : int
            The time horizon for the product in weeks.
        ko_price : float
            The knock-out price.
        guarantee_period : int
            The guaranteed period in weeks during which shares are still accumulated even if KO occurs.
        monitoring_freq : int
            The frequency (in day) of monitoring the settlement price (e.g., 7 days for weekly monitoring).
        gearing : float
            The multiplier for shares per day if the settlement price is below the forward price.
        accumulate_if_barrier : bool, default=True
            Whether to continue accumulation after the knock-out barrier is breached.
        tol : float, default=1e-10
            Tolerance level for numerical calculations to avoid floating-point precision issues.
        path : int, default=10000
            The number of Monte Carlo simulation paths for price modeling.
        antithetic : bool, default=True
            Whether to use antithetic variates for variance reduction in Monte Carlo simulations.
        seed : int, default=0
            Seed for the random number generator to ensure reproducibility.
        """
        self.eq_model = eq_model
        self.yc = yc
        self.t = t
        self.ko_price = ko_price
        self.guarantee_period = guarantee_period
        self.monitoring_freq = monitoring_freq
        self.gearing = gearing
        self.accumulate_if_barrier = accumulate_if_barrier
        self.tol = tol
        self.path = path
        self.antithetic = antithetic
        self.seed = seed
        self.prices = self._simulate_prices()

    def _simulate_prices(self):
        np.random.seed(self.seed)
        z_values = np.random.randn(self.path, self.t)
        prices = np.zeros((self.path * (2 if self.antithetic else 1), self.t + 1))
        prices[:, 0] = self.eq_model.s0
        dt = 1 / 52

        for i in range(self.path):
            for j in range(1, self.t + 1):
                d = (j - 1) * self.monitoring_freq
                r = self.yc.get_r(j * self.monitoring_freq)
                m = np.log(prices[i, j - 1] / (self.eq_model.s0 * np.exp(r * (j - 1) * dt)))
                vol = self.eq_model.get_vol(d, m)
                prices[i, j] = prices[i, j - 1] * np.exp((r - 0.5 * (vol ** 2)) * dt + vol * z_values[i, j - 1] * dt ** 0.5)
                if self.antithetic:
                    prices[i + self.path, j] = prices[i + self.path, j - 1] * np.exp((r - 0.5 * (vol ** 2)) * dt - vol * z_values[i, j - 1] * dt ** 0.5)
        return prices

    def get_pv(self, fp):
        pv = np.zeros(self.prices.shape[0])
        for i in range(self.prices.shape[0]):
            is_ko = False
            for j in range(1, self.t + 1):
                just_ko = False
                pv_t = self.yc.get_df(self.monitoring_freq * j) * self.monitoring_freq * (self.prices[i, j] - fp) * (
                    1 if self.prices[i, j] >= fp else self.gearing)
                if not is_ko and self.prices[i, j] >= self.ko_price:
                    is_ko = True
                    just_ko = True
                if j <= self.guarantee_period:  # we will pay if we are still in guarantee period
                    pv[i] += pv_t
                # we will pay even if barrier is triggered (given self.accumulate_if_barrier)
                elif just_ko and self.accumulate_if_barrier:
                    pv[i] += pv_t
                elif is_ko:
                    break
                else:
                    pv[i] += pv_t
        return np.mean(pv)

    @time_decorator
    def get_fp(self):
        lb = 0
        ub = self.prices[0, 0]
        while self.get_pv(ub) > 0:
            lb = ub
            ub *= 2

        fp = ub
        while np.abs(self.get_pv(fp)) >= self.tol:
            fp = (lb + ub) / 2
            if self.get_pv(fp) < 0:
                ub = fp
            else:
                lb = fp
        return fp