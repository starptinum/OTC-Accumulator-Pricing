#%%
from Accmulator import *
from EQModel import *
from YieldCurve import *
import os
import pandas as pd

#%%
# Inputs
date0 = pd.to_datetime('07/11/2024', format='%d/%m/%Y')
s0 = 2844.78
t = 13
ko_price = 1.1 * s0
guarantee_period = 3
monitoring_freq = 7
gearing = 2

#%%
# Example 1: Pricing with yield curve and Dupire model
futures_input = pd.read_csv(os.path.join(os.getcwd(), 'data/futures_input.csv'), index_col=0, parse_dates=True)
vol_input = pd.read_csv(os.path.join(os.getcwd(), 'data/vol_input.csv'), index_col=0, parse_dates=True)
vol_input.columns = vol_input.columns.astype(float)
yc = YieldCurve_from_Futures(date0=date0, s0=s0, futures_input=futures_input)
# yc.plot_yield_curve()
# yc.plot_discount_factor()
eq_model = DupireModel(date0=date0, s0=s0, yc=yc, vol_input=vol_input)
# eq_model.plot_vol_surface()
pricer = Accumulator_MC(eq_model=eq_model, yc=yc, t=t, ko_price=ko_price, guarantee_period=guarantee_period, monitoring_freq=monitoring_freq, gearing=gearing)
pricer.get_fp() / s0

#%%
# Example 2: Pricing with constant volatility and risk-free rate
r = 0.0982
vol = 0.6731
yc = YieldCurve_Constant(date0=date0, r=r)
# yc.plot_yield_curve()
# yc.plot_discount_factor()
eq_model = BlackScholesConstantVolModel(date0=date0, s0=s0, vol=vol)
# eq_model.plot_vol_surface()
pricer = Accumulator_MC(eq_model=eq_model, yc=yc, t=t, ko_price=ko_price, guarantee_period=guarantee_period, monitoring_freq=monitoring_freq, gearing=gearing)
pricer.get_fp() / s0