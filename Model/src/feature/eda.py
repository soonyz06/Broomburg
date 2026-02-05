import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import scipy.stats as stats
from scipy.stats import chi2
from statsmodels.nonparametric.smoothers_lowess import lowess
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from src.utils.logg import log_plot, log_csv
#Why and What Info

def missing_heatmap(df, SHOW=True):
    fig = plt.figure(figsize=(14, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    log_plot(fig, SHOW)
    
def plot_cat_counts(df, col, ax, **kwargs):
    df[col].value_counts().sort_index().plot(
        kind="bar", edgecolor="black", ax=ax
    )
    ax.set_title(f"Counts of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    return ax

def plot_cat_box(df, col, target, ax, **kwargs):
    grand_median = df[target].median()
    categories = sorted(df[col].dropna().unique())
    grouped_vals = [df.loc[df[col] == c, target].dropna() for c in categories]
    ax.boxplot(grouped_vals, labels=categories)
    ax.axhline(grand_median, color="red", linestyle="--", linewidth=1.5,
               label=f"Grand Median = {grand_median:.2f}")
    ax.set_title(f"{col} vs {target}")
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="best")
    return ax

def plot_cat_violin(df, col, target, ax, **kwargs):
    grand_median = df[target].median()
    categories = sorted(df[col].dropna().unique())
    grouped_vals = [df.loc[df[col] == c, target].dropna() for c in categories]
    parts = ax.violinplot(grouped_vals, vert=True, showmedians=True, **kwargs)
    ax.axhline(grand_median, color="red", linestyle="--", linewidth=1.5,
               label=f"Grand Median = {grand_median:.2f}")
    ax.set_title(f"{col} vs {target}")
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories, rotation=90)
    ax.legend(loc="best")
    return ax

def fd_bins(series):
    x = series.dropna().values
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(x) ** (1/3))
    bins = int((x.max() - x.min()) / bin_width)
    return max(1, bins)

def plot_hist_kde(df, col, ax, **kwargs):
    bins = fd_bins(df[col])
    df[col].dropna().plot(kind="hist", bins=bins, alpha=0.6, density=True, ax=ax)
    df[col].dropna().plot(kind="kde", color="red", ax=ax)
    ax.set_title(f"Histogram + KDE of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    return ax

def plot_qq(df, col, ax, **kwargs):
    stats.probplot(df[col].dropna(), dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot of {col}")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    return ax

def plot_scatter(df,col,target,ax,**kwargs):
    x=df[col].values
    y=df[target].values
    grand_median=df[target].median()
    
    ax.scatter(x,y,alpha=0.3,label="Raw data",**kwargs)
    model=LinearRegression()
    model.fit(x.reshape(-1,1),y)
    y_pred=model.predict(x.reshape(-1,1))
    r2=r2_score(y,y_pred)
    rmse=np.sqrt(mean_squared_error(y,y_pred))
    ax.plot(x,y_pred,color="black",linewidth=2,label=f"OLS: R²={r2:.2f}, RMSE={rmse:.2f}")
    loess_fit=lowess(y,x,frac=0.3)
    ax.plot(loess_fit[:,0],loess_fit[:,1],color="red",linewidth=2,label="LOESS fit")
    ax.axhline(grand_median,color="gray",linestyle="--",linewidth=1.5,label="Median")
    ax.set_title(f"{col} vs {target}")
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    ax.legend()
    return ax

def plot_summary(df, plot_funcs, kwargs_list, figsize=(14,6), SHOW=True):
    nplots = len(plot_funcs)
    fig, axes = plt.subplots(1, nplots, figsize=figsize)
    if nplots == 1:
        axes = [axes]
    for ax, func, kwargs in zip(axes, plot_funcs, kwargs_list):
        func(df, **kwargs, ax=ax)
    plt.tight_layout()
    log_plot(fig, SHOW)

def compute_corr(df, cols, method="pearson"):
    method = method.lower() #["pearson", "spearman", "kendall"]
    if method =="diff":
        pearson_corr = df[cols].corr(method="pearson")
        spearman_corr = df[cols].corr(method="spearman")
        corr = abs(spearman_corr)-abs(pearson_corr)
    else:
        corr = df[cols].corr(method=method)
    return corr

def plot_target_corr(df, feat, method="pearson", SHOW=True):
    cols = feat["num"] 
    target = feat["target"]
    corr_matrix = compute_corr(df, cols + target, method=method)
    corr = corr_matrix[target[0]].drop(target).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(28, 8))
    sns.barplot(x=corr.index, y=corr.values, ax=ax)
    ax.set_title(f"Correlations with {target[0].title()} ({method.title()})")
    ax.set_xlabel("Features")
    ax.set_ylabel(f"{method.title()} Correlations")
    ax.tick_params(axis="x", rotation=90, labelsize=5)
    plt.tight_layout()
    log_plot(fig, SHOW)

def plot_feat_corr(df, cols, limit, method="pearson", SHOW=True):
    corr = compute_corr(df, cols, method=method).fillna(0)
    max_corr_per_feature = corr.abs().replace(1.0, np.nan).max(axis=1)
    top_features = max_corr_per_feature.sort_values(ascending=False).head(limit).index.tolist()
    filtered_corr_top = corr.loc[top_features, top_features]

    fig_top, ax_top = plt.subplots(figsize=(20, 8))
    sns.heatmap(filtered_corr_top, annot=True, cmap="coolwarm", center=0, ax=ax_top)
    ax_top.set_title(f"Top {limit} Features by Strongest {method.title()} Correlation")
    plt.tight_layout()
    log_plot(fig_top, SHOW)

    if limit < 50: return
    bot_features = max_corr_per_feature.sort_values(ascending=True).head(limit).index.tolist()
    filtered_corr_bot = corr.loc[bot_features, bot_features]

    fig_bot, ax_bot = plt.subplots(figsize=(20, 8))
    sns.heatmap(filtered_corr_bot, annot=True, cmap="coolwarm", center=0, ax=ax_bot)
    ax_bit.set_title(f"Bottom {limit} Features by Weakest {method.title()} Correlation")
    plt.tight_layout()
    log_plot(fig_bot, SHOW)
    
def display_nulls(df, cols, n=10):
    null_counts = df[cols].isnull().sum()
    print("\nNull Count")
    print(null_counts[null_counts>0].sort_values(ascending=False).head(n))

def hausman_test(df, entity, target, cols):
    df = df.set_index([entity, "date"])
    X = df[cols]
    
    fe_res = PanelOLS(df[target], X, entity_effects=True).fit() #Fixed Effects (entity effects are correlated with regressor)
    re_res = RandomEffects(df[target], X).fit() #Random Effects (entity effects are uncorrelated with the regressor)
    common_idx = fe_res.params.index.intersection(re_res.params.index)
    beta_diff = fe_res.params[common_idx] - re_res.params[common_idx]
    var_diff = fe_res.cov.loc[common_idx, common_idx] - re_res.cov.loc[common_idx, common_idx]
    stat = float(beta_diff.T @ np.linalg.inv(var_diff) @ beta_diff)
    df_h = len(beta_diff)
    pval = 1 - chi2.cdf(stat, df_h)    
    print(f"\nHausman Test Statistic for {entity}: {stat:.2f}") #H0: RE is better 
    print(f"Degrees of freedom: {df_h}")
    print(f"p-value: {pval:.4f}")
    return stat, pval

def plot_residual_diag_reg(y_pred, residuals, target="Target"): #redudant but whatever
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(y_pred, residuals, alpha=0.3)
    ax[0].axhline(0, color='r', linestyle='--', linewidth=2)
    ax[0].set_title(f"Residuals vs Fitted for {target}")
    ax[0].set_xlabel("Fitted Values")
    ax[0].set_ylabel("Residuals")
    
    stats.probplot(np.asarray(residuals), dist="norm", plot=ax[1])
    ax[1].set_title(f"QQ Plot of Residuals for {target}")
    ax[1].set_xlabel("Theoretical Quantiles")
    ax[1].set_ylabel("Sample Quantiles")
    plt.tight_layout()
    return fig

def plot_influence_diag_reg(cooks_d, vif_df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].stem(range(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
    ax[0].set_title("Cook's Distance")
    ax[0].set_xlabel("Observation Index")
    ax[0].set_ylabel("Cook's Distance")
    
    ax[1].bar(vif_df["feature"], vif_df["VIF"], color="skyblue")
    ax[1].set_title("Variance Inflation Factors")
    ax[1].set_ylabel("VIF")
    ax[1].tick_params(axis='x', labelsize=5, rotation=90)
    plt.tight_layout()
    return fig

def plot_regression(df, cols, target, figsize=(14, 6), SHOW=True): #for inference not prediction
    data = df[cols + [target]]
    X = sm.add_constant(data[cols])
    y = data[target]
    model = sm.OLS(y, X).fit(cov_type="HC3") #HAC: autocorrelation 
    y_pred = model.fittedvalues
    residuals = model.resid
    
    coef = model.params
    conf_int = model.conf_int(alpha=0.05)
    p_val = model.pvalues
    results_df = pd.DataFrame({
        "feature": X.columns,  
        "coef": coef.values,
        #"lower_ci": conf_int[0].values,
        #"upper_ci": conf_int[1].values,
        "p_val": p_val.values,
    }).round(4).sort_values(by="p_val")
    cooks_d, pvals = model.get_influence().cooks_distance
    vif_df = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).round(4).sort_values(by="VIF", ascending=False)
    vif_df = vif_df[vif_df["feature"] != "const"]

    fig1, fig2 = None, None
    if SHOW:
        fig1 = plot_residual_diag_reg(y_pred, residuals, target)
        fig2 = plot_influence_diag_reg(cooks_d, vif_df)
        log_plot(fig1, SHOW)
        log_plot(fig2, SHOW)
        log_csv(results_df, "regression_param_results.csv")
    print(model.summary())
    print(f"\nAdjusted R-squared: {model.rsquared_adj:.2f}")
    print(f"F-statistic: {model.fvalue:.2f}")
    return model
    #AIC/BIC: Lower is better (Preference: AIC~Complex, BIC~Simple)
    #---Residuals---
    #Skewness: normal=0
    #JB: H0 of normally distributed 
    #Kurtosis: normal=3 (laptokurtic and platykurtic)
    #DW: no autocorrelation =2

def md_qq_plot(df, SHOW=True):  
    mean = df.mean().values
    cov = df.cov().values
    inv_cov = np.linalg.inv(cov)
    md = np.sort(df.apply(lambda row: (row.values - mean) @ inv_cov @ (row.values - mean).T, axis=1))
    chi2_quantiles = chi2.ppf((np.arange(1, len(md)+1)-0.5)/len(md), df=df.shape[1])

    fig=plt.figure(figsize=(6,6))
    plt.scatter(chi2_quantiles, md, alpha=0.7)
    plt.plot([0, max(chi2_quantiles)], [0, max(chi2_quantiles)], color="red")
    plt.xlabel("Chi-square quantiles")
    plt.ylabel("Mahalanobis distances")
    plt.title("Q-Q Plot for Multivariate Normality")
    log_plot(fig, SHOW)
    
class Custom_EDA(): #filter by cat and choose target
    def __init__(self, df, feat, top_k=10):
        self.df = df
        self.feat = feat
        self.top_k = top_k
        self.target = self.feat["target"][0]

    def set_feat(self, feat):
        self.feat = feat
        return self
    def set_target(self, target):
        if not isinstance(target, list):
            target = [target]
        self.target = target[0]
        self.feat["target"] = target
        return self

    def md_qq_plot(self, features=None, SHOW=True):
        if features is None: features = self.feat["num"]
        md_qq_plot(self.df[features], SHOW=SHOW)
        
    def hausman_test(self, limit=None, target=None):
        if limit is None: limit = self.top_k
        if target is None: target = self.target
        for col in tqdm(self.feat["cat"][:limit], desc="Hausman Test"): #heterogeneity
            hausman_test(self.df, col, target, self.feat["num"])
        print("\n")
        return self
    
    def categorical_dist(self, limit=None, target=None, SHOW=True):
        if target is None: target = self.target
        plot_funcs = [plot_cat_counts, plot_cat_violin]
        
        for col in tqdm(self.feat["cat"][:limit], desc="Categorical"): 
            params = [{"col": col, "target": target}]*len(plot_funcs)
            plot_summary(self.df, plot_funcs, params, SHOW=SHOW)
        print("\n")
        return self
    
    def numerical_dist(self, limit=None, target=None, SHOW=True):
        if limit is None: limit = self.top_k
        if target is None: target = self.target
        plot_funcs = [plot_hist_kde, plot_qq, plot_scatter]
        
        for col in tqdm(self.feat["num"][:limit], desc="Numerical"):
            params = [{"col": col, "target": target}]*len(plot_funcs)
            plot_summary(self.df, plot_funcs, params, SHOW=SHOW)
        print("\n")
        return self
    
    def target_dist(self, target=None, SHOW=True):
        if target is None: target = self.target
        plot_funcs = [plot_hist_kde, plot_qq]
        
        params = [{"col": target, "target": target}]*len(plot_funcs)
        plot_summary(self.df, plot_funcs, params, SHOW=SHOW)
        print("\n")
        return self
    
    def feature_corr(self, method="pearson", SHOW=True):
        plot_feat_corr(self.df, self.feat["num"], limit=self.top_k, method=method, SHOW=SHOW)
        return self
    
    def target_corr(self, method="pearson", SHOW=True):
        plot_target_corr(self.df, self.feat, method=method, SHOW=SHOW)
        return self
    
    def regression(self, SHOW=True):
        plot_regression(self.df, self.feat["num"], self.target, SHOW=SHOW)
        return self

    def describe(self, target=None): #pretransformed for interpretation
        if target is None: target = self.target
        print(self.df[target].describe().round(2))
        return self

#IC correlation between predicted vs actual values
#MI how much knowing one variable reduces uncertainty of anothers
