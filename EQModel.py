from scipy.optimize import minimize
from util import time_decorator
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import pandas as pd


class EQModel:
    def __init__(self, date0, s0):
        self.date0 = date0
        self.s0 = s0

    def get_vol(self, d, m):
        raise NotImplementedError("Subclasses must implement the `get_vol` method.")

    @time_decorator
    def plot_vol_surface(self, m_grid=None, d_grid=None):
        """
        Plot the volatility surface.

        Parameters
        ----------
        m_grid : array-like
            A range of moneyness values.
        d_grid : array-like
            A range of days.
        """
        if m_grid is None:
            m_grid = np.linspace(-0.5, 0.5)
        if d_grid is None:
            d_grid = np.linspace(5, 100)
        vol_surface = pd.Series(index=pd.MultiIndex.from_product([d_grid, m_grid], names=['d', 'm']))

        for d in d_grid:
            for m in m_grid:
                vol_surface.loc[(d, m),] = self.get_vol(d=d, m=m)

        x_vals = vol_surface.index.get_level_values('d').unique()
        y_vals = vol_surface.index.get_level_values('m').unique()
        z_vals = vol_surface.values.reshape(len(x_vals), len(y_vals)).T

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_vals, y_vals)
        surface = ax.plot_surface(X, Y, z_vals, cmap='viridis')
        fig.colorbar(surface)
        ax.set_xlabel(f"Days from {self.date0.strftime('%Y-%m-%d')}")
        ax.set_ylabel("Moneyness")
        plt.show()


class DupireModel(EQModel):
    """
    A class to represent the Dupire Local Volatility Model.

    Parameters
    ----------
    date0 : pd.Timestamp
        The date to price the derivative.
    s0 : float
        Asset price at time date0.
    yc : YieldCurve
        Yield curve object
    vol_input : pd.DataFrame
        A pd.DataFrame representing the implied volatility input.
        The data should be in the form of (pd.Timestamp, Strike).
    days_a_year : int, optional
        The number of trading days in a year. Default is 365.
    n_trial : int, optional
        The number of initialization in the fitting of implied volatility curve. Default is 10.
    seed : int, optional
        The seed used to generate the initialization for the fitting of implied volatility curve. Default is 0.
    tol : float, optional
        Tolerance level for optimization (default: 1e-10).
    """

    def __init__(self, date0, s0, yc, vol_input, days_a_year=365, n_trial=10, seed=0, tol=1e-10):
        super().__init__(date0, s0)
        self.yc = yc
        self.vol_input = vol_input
        self.days_a_year = days_a_year
        self.n_trial = n_trial
        self.seed = seed
        self.tol = tol
        self.res_params = self._fit_iv_curve()
        self.fitted_lv = {}

    @staticmethod
    def _parametric_curve(param, m, p0):
        """
        Parametric model for the implied volatility curve.

        Parameters
        ----------
        param : array-like
            Parameters for the volatility curve (delta, gamma, kappa).
        m : float or array-like
            Moneyness is defined as ln(K / F).
        p0 : tuple
            Fixed point in the curve.

        Returns
        -------
        variance : float or array-like
            Implied variance corresponding to the moneyness.
        """
        wings_func = lambda y: np.tanh(param[2] * y) / param[2]
        wings = wings_func(m - p0[0])
        return p0[1] ** 2 + param[0] * wings + 0.5 * param[1] * wings ** 2

    def _fit_iv_curve(self):
        """
        Fit the implied volatility curve for each date in the data using optimization.

        Returns
        -------
        res_params : pd.DataFrame
            Fitted parameters for all dates in the data.
        """
        np.random.seed(self.seed)
        res_params = pd.DataFrame(columns=['delta', 'gamma', 'kappa', 'moneyness0', 'sig0'], dtype=float)
        for date in self.vol_input.index:
            iv_data = self.vol_input.loc[date].dropna()
            iv_data.index = np.log(
                iv_data.index / self.s0 * self.yc.get_df((date - self.date0).days))
            m = iv_data.index
            m0 = m[np.abs(m).argmin()]  # Moneyness closest to zero
            sig0 = iv_data.loc[m0]
            p0 = (m0, sig0)
            # fitting of implied volatility curve
            x, loss = None, None
            for i in range(self.n_trial):
                x0 = np.random.rand(3)
                while self._parametric_curve(param=x0, m=m, p0=p0).min() < 0:
                    x0 = np.random.rand(3)
                res = minimize(
                    lambda y: np.mean(
                        np.power(iv_data - np.sqrt(self._parametric_curve(param=y, m=m, p0=p0)), 2)), x0,
                    method='BFGS', tol=self.tol)
                if loss is None or loss > res.fun:
                    loss = res.fun
                    x = res.x
            res_params.loc[(date - self.date0).days, ['delta', 'gamma', 'kappa']] = x
            res_params.loc[(date - self.date0).days, ['moneyness0', 'sig0']] = p0
        return res_params

    def _plot_fitted_iv_curve(self):
        """
        Plot all fitted implied volatility curves for each date in the data.
        """
        for date in self.vol_input.index:
            iv_data = self.vol_input.loc[date].dropna()
            iv_data.index = np.log(
                iv_data.index / self.s0 * self.yc.get_df((date - self.date0).days))
            m = iv_data.index
            plt.scatter(m, iv_data)
            param = self.res_params.loc[(date - self.date0).days, ['delta', 'gamma', 'kappa']].values
            p0 = self.res_params.loc[(date - self.date0).days, ['moneyness0', 'sig0']].values
            plt.plot(m, np.sqrt(self._parametric_curve(param=param, m=m, p0=p0)))
            plt.title(f'Implied Volatility Surface at {date.strftime("%Y-%m-%d")}')
            plt.xlabel(r'Moneyness')
            plt.ylabel(r'Implied Volatility')
            plt.show()

    def _get_iv(self, d, m):
        """
        Get the implied volatility for a given day and moneyness.

        Parameters
        ----------
        d : int
            Day
        m : float
            Moneyness.

        Returns
        -------
        iv : float
            Implied volatility.
        """

        def _helper(d):
            delta, gamma, kappa, m0, sig0 = self.res_params.loc[d]
            return np.sqrt(self._parametric_curve(param=(delta, gamma, kappa), m=m, p0=(m0, sig0)))

        if d in self.res_params.index:
            return _helper(d)

        # to find the points using linear interpolation
        previous_day = self.res_params.loc[:d].index[-1] if len(self.res_params.loc[:d].index) > 0 else None
        next_day = self.res_params.loc[d:].index[0] if len(self.res_params.loc[d:].index) > 0 else None

        if previous_day is None:
            previous_day, next_day = self.res_params.index[:2]
        elif next_day is None:
            previous_day, next_day = self.res_params.index[-2:]

        previous_sigma, next_sigma = _helper(previous_day) ** 2, _helper(next_day) ** 2
        return np.sqrt(((next_sigma * next_day - previous_sigma * previous_day) / (next_day - previous_day) * (
                d - previous_day) + previous_sigma * previous_day) / d)

    def _fit_lv(self, d, m_grid=np.linspace(-0.5, 0.5, 200)):
        """
        Fit the local volatility for given moneyness and day.

        Parameters
        ----------
        d : int
            Day
        m_grid : array-like, optional
            A range of moneyness values
        """

        def _total_volatility(x):
            return x[0] * (self._get_iv(d=x[0] * self.days_a_year, m=x[1]) ** 2)

        grad_fn = nd.Gradient(_total_volatility)
        hessian_fn = nd.Hessian(_total_volatility)

        def _helper(m):
            x = (d / self.days_a_year, m)
            w, grad, hessian_matrix = _total_volatility(x), grad_fn(x), hessian_fn(x)
            return grad[0] / (1 - x[1] * grad[1] / w + 0.25 * (-0.25 - 1 / w + (x[1] / w) ** 2) * (grad[1] ** 2) + 0.5 *
                              hessian_matrix[1][1])

        # Interpolation if local volatility is negative
        v_grid = pd.DataFrame(index=m_grid, columns=['v'], dtype=float)
        v_grid = v_grid.apply(lambda x: _helper(x.name), axis=1)
        v_grid[v_grid.lt(0)] = np.nan
        self.fitted_lv[d] = np.sqrt(v_grid)
        self.fitted_lv[d] = self.fitted_lv[d].interpolate(method='index', limit_direction='both')

    def get_vol(self, d, m):
        """
        Get the local volatility for a given day and moneyness.

        Parameters
        ----------
        d : int
            Day
        m : float
            Moneyness.

        Returns
        -------
        lv : float
            Local volatility.
        """

        if d == 0:
            first_row = self.vol_input.iloc[0].dropna()
            closest_col = first_row.index[np.abs(first_row.index - self.s0).argmin()]
            return first_row[closest_col] if not pd.isna(first_row[closest_col]) else np.nan

        if d not in self.fitted_lv.keys():
            self._fit_lv(d)

        self.fitted_lv[d] = self.fitted_lv[d].reindex(self.fitted_lv[d].index.union([m]))
        self.fitted_lv[d] = self.fitted_lv[d].interpolate(method='index', limit_direction='both')
        return self.fitted_lv[d].loc[m]


class BlackScholesImpliedVolModel(EQModel):
    def __init__(self, date0, s0, yc, vol_input, n_trial=10, seed=0, tol=1e-10):
        super().__init__(date0, s0)
        self.yc = yc
        self.vol_input = vol_input
        self.n_trial = n_trial
        self.seed = seed
        self.tol = tol
        self.res_params = self._fit_iv_curve()

    @staticmethod
    def _parametric_curve(param, m, p0):
        """
        Parametric model for the implied volatility curve.

        Parameters
        ----------
        param : array-like
            Parameters for the volatility curve (delta, gamma, kappa).
        m : float or array-like
            Moneyness is defined as ln(K / F).
        p0 : tuple
            Fixed point in the curve.

        Returns
        -------
        variance : float or array-like
            Implied variance corresponding to the moneyness.
        """
        wings_func = lambda y: np.tanh(param[2] * y) / param[2]
        wings = wings_func(m - p0[0])
        return p0[1] ** 2 + param[0] * wings + 0.5 * param[1] * wings ** 2

    def _fit_iv_curve(self):
        """
        Fit the implied volatility curve for each date in the data using optimization.

        Returns
        -------
        res_params : pd.DataFrame
            Fitted parameters for all dates in the data.
        """
        np.random.seed(self.seed)
        res_params = pd.DataFrame(columns=['delta', 'gamma', 'kappa', 'moneyness0', 'sig0'], dtype=float)
        for date in self.vol_input.index:
            iv_data = self.vol_input.loc[date].dropna()
            iv_data.index = np.log(
                iv_data.index / self.s0 * self.yc.get_df((date - self.date0).days))
            m = iv_data.index
            m0 = m[np.abs(m).argmin()]  # Moneyness closest to zero
            sig0 = iv_data.loc[m0]
            p0 = (m0, sig0)
            # fitting of implied volatility curve
            x, loss = None, None
            for i in range(self.n_trial):
                x0 = np.random.rand(3)
                while self._parametric_curve(param=x0, m=m, p0=p0).min() < 0:
                    x0 = np.random.rand(3)
                res = minimize(
                    lambda y: np.mean(
                        np.power(iv_data - np.sqrt(self._parametric_curve(param=y, m=m, p0=p0)), 2)), x0,
                    method='BFGS', tol=self.tol)
                if loss is None or loss > res.fun:
                    loss = res.fun
                    x = res.x
            res_params.loc[(date - self.date0).days, ['delta', 'gamma', 'kappa']] = x
            res_params.loc[(date - self.date0).days, ['moneyness0', 'sig0']] = p0
        return res_params

    def _plot_fitted_iv_curve(self):
        """
        Plot all fitted implied volatility curves for each date in the data.
        """
        for date in self.vol_input.index:
            iv_data = self.vol_input.loc[date].dropna()
            iv_data.index = np.log(
                iv_data.index / self.s0 * self.yc.get_df((date - self.date0).days))
            m = iv_data.index
            plt.scatter(m, iv_data)
            param = self.res_params.loc[(date - self.date0).days, ['delta', 'gamma', 'kappa']].values
            p0 = self.res_params.loc[(date - self.date0).days, ['moneyness0', 'sig0']].values
            plt.plot(m, np.sqrt(self._parametric_curve(param=param, m=m, p0=p0)))
            plt.title(f'Implied Volatility Surface at {date.strftime("%Y-%m-%d")}')
            plt.xlabel(r'Moneyness')
            plt.ylabel(r'Implied Volatility')
            plt.show()

    def get_vol(self, d, m):
        """
        Get the implied volatility for a given day and moneyness.

        Parameters
        ----------
        d : int
            Day
        m : float
            Moneyness.

        Returns
        -------
        iv : float
            Implied volatility.
        """

        def _helper(d):
            delta, gamma, kappa, m0, sig0 = self.res_params.loc[d]
            return np.sqrt(self._parametric_curve(param=(delta, gamma, kappa), m=m, p0=(m0, sig0)))

        if d in self.res_params.index:
            return _helper(d)

        # to find the points using linear interpolation
        previous_day = self.res_params.loc[:d].index[-1] if len(self.res_params.loc[:d].index) > 0 else None
        next_day = self.res_params.loc[d:].index[0] if len(self.res_params.loc[d:].index) > 0 else None

        if previous_day is None:
            previous_day, next_day = self.res_params.index[:2]
        elif next_day is None:
            previous_day, next_day = self.res_params.index[-2:]

        previous_sigma, next_sigma = _helper(previous_day) ** 2, _helper(next_day) ** 2
        return np.sqrt(((next_sigma * next_day - previous_sigma * previous_day) / (next_day - previous_day) * (
                d - previous_day) + previous_sigma * previous_day) / d)


class BlackScholesConstantVolModel(EQModel):
    def __init__(self, date0, s0, vol):
        super().__init__(date0, s0)
        self.vol = vol

    def get_vol(self, d, m):
        return self.vol