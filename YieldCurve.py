from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
import numpy as np


class YieldCurve:
    """
    Base class for yield curves.
    """
    def __init__(self, date0, days_a_year=365):
        self.date0 = date0
        self.days_a_year = days_a_year

    def get_r(self, d):
        raise NotImplementedError("Subclasses must implement `get_r`.")

    def get_df(self, d):
        """
        Compute the discount factor for a given day.
        """
        return np.exp(-self.get_r(d) * d / self.days_a_year)

    def plot_yield_curve(self, d_grid=None):
        """
        Plot the yield curve (zero rates vs. days).
        """
        if d_grid is None:
            d_grid = np.linspace(0, self.days_a_year * 5, 200)
        plt.plot(d_grid, self.get_r(d_grid), '-')
        plt.xlabel(f'Days from {self.date0.strftime("%Y-%m-%d")}')
        plt.ylabel('Zero Rates')
        plt.title('Yield Curve')
        plt.show()

    def plot_discount_factor(self, d_grid=None):
        """
        Plot the discount factor curve.
        """
        if d_grid is None:
            d_grid = np.linspace(0, self.days_a_year * 5, 200)
        plt.plot(d_grid, self.get_df(d_grid), '-')
        plt.xlabel(f'Days from {self.date0.strftime("%Y-%m-%d")}')
        plt.ylabel('Discount Factor')
        plt.title('Discount Factor Curve')
        plt.show()


class YieldCurve_from_Futures(YieldCurve):
    """
    Yield curve derived from futures prices.
    """
    def __init__(self, date0, s0, futures_input, interpolate='cubic', days_a_year=365):
        super().__init__(date0, days_a_year)
        self.s0 = s0
        self.futures_input = futures_input
        self.days_from_t0 = (self.futures_input.index - date0).days
        self.known_yield = np.log(self.futures_input['price'].values / s0) / self.days_from_t0 * self.days_a_year

        # Select interpolation method
        if interpolate == 'linear':
            self.interpolator = interp1d(self.days_from_t0, self.known_yield, kind='linear', fill_value="extrapolate")
        elif interpolate == 'cubic':
            self.interpolator = CubicSpline(self.days_from_t0, self.known_yield, bc_type='natural', extrapolate=True)
        else:
            raise ValueError(f"Unsupported interpolation type: {interpolate}")

    def get_r(self, d):
        """
        Get the zero rate for a given day using interpolation.
        """
        return self.interpolator(d)

    def plot_yield_curve(self, d_grid=None):
        """
        Plot the yield curve with known zero rates as points.
        """
        if d_grid is None:
            d_grid = np.linspace(0, self.days_from_t0.max(), 200)
        plt.plot(self.days_from_t0, self.get_r(self.days_from_t0), 'o', label='Known Zero Rates')
        plt.plot(d_grid, self.get_r(d_grid), '-', label='Interpolated Curve')
        plt.xlabel(f'Days after {self.date0.strftime("%Y-%m-%d")}')
        plt.ylabel('Zero Rates')
        plt.title('Yield Curve from Futures')
        plt.legend()
        plt.show()

    def plot_discount_factor(self, d_grid=None):
        """
        Plot the discount factor curve.
        """
        if d_grid is None:
            d_grid = np.linspace(0, self.days_from_t0.max(), 200)
        plt.plot(self.days_from_t0, self.get_df(self.days_from_t0), 'o', label='Known Discounted Factors')
        plt.plot(d_grid, self.get_df(d_grid), '-', label='Interpolated Curve')
        plt.xlabel(f'Days from {self.date0.strftime("%Y-%m-%d")}')
        plt.ylabel('Discount Factor')
        plt.title('Discount Factor Curve')
        plt.show()


class YieldCurve_Constant(YieldCurve):
    """
    Yield curve with a constant zero rate.
    """
    def __init__(self, date0, r, days_a_year=365):
        super().__init__(date0, days_a_year)
        self.r = r

    def get_r(self, d):
        """
        Get the constant zero rate for a given day.
        """
        return self.r if np.isscalar(d) else self.r * np.ones(d.shape)