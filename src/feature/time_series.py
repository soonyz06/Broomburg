import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose

from src.utils.logg import log_plot


def plot_periodogram(ts, fs=1.0):
    ts_clean = ts.dropna() if hasattr(ts, "dropna") else ts[~np.isnan(ts)]
    freqs, power = periodogram(ts_clean, fs=fs)
    freqs, power = freqs[1:], power[1:]
    fig = plt.figure(figsize=(10,6))
    plt.semilogy(freqs, power)
    plt.title("Periodogram")
    plt.xlabel("Frequency")
    plt.ylabel("Spectral Density")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    log_plot(fig)

def stationarity_tests(ts, regression='c', nlags='auto'):
    ts_clean = ts.dropna() if hasattr(ts, "dropna") else ts[~np.isnan(ts)]
    adf_res = adfuller(ts_clean, autolag='AIC')
    kpss_res = kpss(ts_clean, regression=regression, nlags=nlags)
    return {
        "ADF": {
            "statistic": adf_res[0],
            "pvalue": adf_res[1],
            "lags": adf_res[2],
            "nobs": adf_res[3],
            "crit_values": adf_res[4]
        },
        "KPSS": {
            "statistic": kpss_res[0],
            "pvalue": kpss_res[1],
            "lags": kpss_res[2],
            "crit_values": kpss_res[3]
        }
    }
    #ADF: H0 is non-stationary
    #KPSS: H0 is stationary
    #Cointegratin: Spread is stationary, where spread = linear combination of prices (index arb, pairs trading, etc)

def plot_acf_pacf(ts, freq=30, alpha=0.05):
    #+ve: momentum, -ve: reversion, else: no pattern (at alpha level)
    fig, axes = plt.subplots(1, 2, figsize=(18,6))
    plot_acf(ts.dropna(), lags=freq, ax=axes[0], alpha=alpha)
    axes[0].set_title(f"Autocorrelation Function (ACF) - {int((1-alpha)*100)}% CI")

    plot_pacf(ts.dropna(), lags=freq, ax=axes[1], method="ywm", alpha=alpha)
    axes[1].set_title(f"Partial Autocorrelation Function (PACF) - {int((1-alpha)*100)}% CI")
    plt.tight_layout()
    log_plot(fig)

def plot_stl_decom(ts, period):
    stl = STL(ts, period=period, robust=True)
    res = stl.fit()
    fig = res.plot()
    fig.set_size_inches(10, 8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    log_plot(fig)

    x = ts.dropna()
    t = res.trend.reindex(x.index)
    s = res.seasonal.reindex(x.index)
    r = res.resid.reindex(x.index)

    x = x - x.mean()
    t = t - t.mean()
    s = s - s.mean()
    r = r - r.mean()

    vx = np.var(x)
    vt = np.var(t)
    vs = np.var(s)
    vr = np.var(r)
    overlap = 1 - (vt + vs + vr) / vx

    shares = {"trend": vt / vx, "seasonal": vs / vx, "residual": vr / vx, "overlap": overlap}
    fig = plt.figure(figsize=(6,4))
    plt.bar(list(shares.keys()), list(shares.values()), color=["orange","teal","red","gray"])
    plt.ylim(0, 1)
    plt.ylabel("Variance share")
    plt.title("STL Variance Decomposition")
    log_plot(fig)
    return shares

def plot_seasonal_decom(ts, period, model_type):
    decomposition = seasonal_decompose(ts, model=model_type, period=period, extrapolate_trend=0)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    log_plot(fig)

def plot_bands(ts, window=20, alpha=0.05):
    rolling_mean = ts.rolling(window).mean()
    rolling_std = ts.rolling(window).std()
    z = norm.ppf(1 - alpha/2)
    upper_band = rolling_mean + z * rolling_std
    lower_band = rolling_mean - z * rolling_std

    fig = plt.figure(figsize=(10, 6))
    plt.plot(ts.index, ts, color="blue", alpha=0.5)
    plt.plot(rolling_mean.index, rolling_mean, color="orange")
    plt.plot(upper_band.index, upper_band, color="green", linestyle="--")
    plt.plot(lower_band.index, lower_band, color="red", linestyle="--")
    plt.title(f"Returns Bands (alpha={alpha})")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.grid(True)
    log_plot(fig)


##fourier, corr, ta, volume ~time series specific 
