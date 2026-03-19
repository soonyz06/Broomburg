import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import t
from arch import arch_model

class Risk:
    def close_to_close_vol(self, df, window=21, annualised=False, periods_per_year=252):
        closes = np.array(df["close"])
        log_returns = np.diff(np.log(closes))
        df_roll = pl.DataFrame({"date": df["date"][1:], "log_ret": log_returns})
        vol = df_roll["log_ret"].rolling_std(window_size=window, ddof=1) #per interval vol & bessel correction
        if annualised:
            vol = vol * np.sqrt(periods_per_year) #annualized vol ~ variance is additive
        vol = pl.DataFrame({"date": df_roll["date"], "vol": vol}).drop_nulls()
        return vol

    def yang_zhang_vol(self, df, annualised=False, window=21, periods_per_year=252):
        _open = np.array(df["open"])
        _high = np.array(df["high"])
        _low = np.array(df["low"])
        _close = np.array(df["close"])
        alpha = 1.34 #1.331 when drift is 0, but the effects of drift on α is minor
        
        norm_open = np.log(_open[1:] / _close[:-1]) #o
        norm_high = np.log(_high[1:] / _open[1:]) #u
        norm_low = np.log(_low[1:] / _open[1:]) #d
        norm_close = np.log(_close[1:] / _open[1:]) #c
        rs = norm_high*(norm_high - norm_close) + norm_low*(norm_low - norm_close) #0 when one-direction move (u=c, d=c, u=0, d=0)

        df_roll = pl.DataFrame({"date": df["date"][1:], "norm_open": norm_open, "norm_close": norm_close, "rs": rs })
        df_roll = df_roll.with_columns(pl.all().exclude("date").cast(pl.Float64))

        #fT - Before opening (opening jump as an unobservable continuous price movement)
        #(1-f)T - Trading interval (fraction f of period T) 
        var_o = df_roll["norm_open"].rolling_var(window_size=window, ddof=1) #E[V_o] = σ^2f
        var_c = df_roll["norm_close"].rolling_var(window_size=window, ddof=1) #E[V_c] = σ^2(1-f)
        var_rs = df_roll["rs"].rolling_mean(window_size=window) #E[V_rs] = σ^2(1-f)

        n = window
        k = (alpha - 1) / (alpha + (n+1)/(n-1)) #k can never reach 0 or 1, so neither V_cc or V_rs alone can find the minimum variance
        total_var = var_o + k*var_c + (1-k)*var_rs #E[V] = E[V_o] + kE[V_c] + (1-k)E[V_rs] = σ^2, as f + (1-f) = 1
        vol = total_var.sqrt()
        if annualised:
            vol = vol * np.sqrt(periods_per_year) #unbiased and independent of both drift(μ) and opening jump(f)
        vol = pl.DataFrame({"date": df_roll["date"], "vol": vol}).drop_nulls()
        return vol

    def garch_vol(self, df): 
        #conditional variance (h_t) = unconditional variance (w) + q ∑ a*past squared shicks (ε^2) + p ∑ b*past conditional variance (h_t)
        #E[h_t] = sigma^2 =  w/(1-(a+b)), so ∑a + ∑b < 1 for it to be stationary (else no unconditional variance)
        #QMLE using pseudo-correlation matrix
        
        log_returns = df["log_returns"].to_numpy()
        model = arch_model(log_returns, vol="GARCH", p=1, q=1, dist="t") #Univariate t-GARCH(p=1, q=1) 
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=1)
        vol = forecast.variance.iloc[-1, 0]**0.5
        return vol, res.params
            
    def calc_VaR(self, vol, nu, mu=0, alpha=0.01): 
        critical = t.ppf(alpha, df=nu)
        VaR = -(mu + critical * vol)
        return VaR
    #ES and 10 day horizon (MC and deterministic)
    #Vol-targetting (target/actual), Vol arbitrage, Portfolio optimisation (DCC)
