import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import polars as pl
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, average_precision_score
from scipy.stats import kendalltau
from sklearn.inspection import partial_dependencem permutation_importance
from PyALE import ale
import shap

from src.utils.logg import log_plot, log_csv


def results_summary(y_test, y_pred, metrics=None):
    if metrics is None:
        metrics = ["r2", "rmse", "mae", "corr"]
    results = {"r2": r2_score(y_test, y_pred) * 100}
    for m in metrics:
        if m == "mae":
            results["mae"] = mean_absolute_error(y_test, y_pred) * 100
        elif m == "rmse":
            results["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred)) * 100
        elif m == "rmsle":
            results["rmsle"] = np.sqrt(mean_squared_log_error(y_test, y_pred)) * 100 #>-1
        elif m == "mape":
            results["mape"] = mean_absolute_percentage_error(y_test, y_pred) * 100
        elif m == "corr":
            tau, _ = kendalltau(y_test, y_pred)
            results["kendall"] = tau * 100
        elif m == "map":
            results["map"] = average_precision_score(y_test, y_pred) * 100
        #rank corr, classification
    for k, v in results.items():
        print(f"{k.upper()}: {v:.1f}%")
    return results

def get_importances(name, features, importances):
    importance_df = pl.DataFrame({
        "feature": features,
        "importance": np.round(importances, 4)
    }).sort("importance", descending=True)
    print(importance_df.head(10))
    return importance_df

def get_permutation_importances(name, model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=1)
    importance_df = pl.DataFrame({
        "feature": X.columns,
        "importance_mean": np.round(result.importances_mean, 4),
        "importance_std": np.round(result.importances_std, 4)
    }).sort("importance_mean", descending=True)
    log_csv(importance_df, f"{name}_permutation_importances.csv")
    return importance_df

def get_global_SHAP(name, features, shap_values):
    shap_df = pl.DataFrame({
        "Feature": features,
        "Global_SHAP": np.round(np.abs(shap_values).mean(axis=0), 4)
    }).sort("Global_SHAP", descending=True)
    log_csv(shap_df, f"{name}_shap.csv")
    return shap_df
    
def plot_fitted_diag(name, y_test, y_pred, SHOW=False):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    low = min(y_test.min(), y_pred.min())
    high = max(y_test.max(), y_pred.max())
    ax.plot([low, high], [low, high], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual')
    ax.grid(True)
    log_plot(fig, SHOW)    

def plot_resid_diag(name, y_test, y_pred, SHOW=False):
    residuals = y_test - y_pred
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].scatter(y_pred, residuals, alpha=0.6)
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].set_xlabel("Predicted values")
    ax[0].set_ylabel("Residuals")
    ax[0].set_title("Residuals vs Predictions")
    
    stats.probplot(np.asarray(residuals), dist="norm", plot=ax[1])
    ax[1].set_title(f"QQ Plot of Residuals")
    ax[1].set_xlabel("Theoretical Quantiles")
    ax[1].set_ylabel("Sample Quantiles")
    
    print(f"Residual mean: {np.mean(residuals)*100:.2f}%")
    print(f"Residual std: {np.std(residuals)*100:.2f}%")
    log_plot(fig, SHOW)    
    
def plot_learning_curve(name, evals_result, metrics=["rmse", "mae"], SHOW=False):
    if evals_result is None:
        return
    datasets = list(evals_result.keys())
    for metric in metrics:
        fig, ax = plt.subplots()
        for ds in datasets:
            values = evals_result[ds][metric]
            epochs = range(1, len(values) + 1)  # define epochs
            ax.plot(epochs, values, label=f"{ds} {metric.upper()}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.set_xticks(epochs)  # now epochs is defined
        log_plot(fig, SHOW)

def get_live_pred(name, symbols, y_pred, shap_values, features):
    live_df = pl.DataFrame(
        {"symbol": symbols,
         "prediction": np.round(y_pred, 4),
         **{f"{feat}": np.round(shap_values[:, i], 4) for i, feat in enumerate(features)}
    }).sort("prediction", descending=True)
    log_csv(live_df, f"{name}_live.csv")
    return live_df

def plot_dependence_diag(name, features, model, X_set, shap_values, SHOW=False):
    for f in features:
        fig = plt.figure(figsize=(21, 5), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1]) 

        ax_pdp = fig.add_subplot(gs[0,0])
        ax_ale = fig.add_subplot(gs[1,0])
        ax_shap = fig.add_subplot(gs[:,1])  

        pd_results = partial_dependence(model, X_set, features=[f], kind="both", grid_resolution=50)
        grid = pd_results["grid_values"][0]
        pdp  = pd_results["average"][0]
        ice  = pd_results["individual"][0]
        ax_pdp.plot(grid, pdp, color="red", linewidth=2)   #PDP: Average effect of a feature on predictions 
        ax_pdp.plot(grid, ice.T, color="blue", alpha=0.1)  #ICE: Effect of a feature per sample (reveals heterogeneity)
        ax_pdp.set_title(f"PDP + ICE", fontsize=8)
        ax_pdp.set_xlabel(f, fontsize=6)
        ax_pdp.set_ylabel("Prediction (average)", fontsize=6)
        ax_pdp.tick_params(axis="both", labelsize=8)
        ax2 = ax_pdp.twinx()
        ax2.hist(X_set[f], bins=30, color="gray", alpha=0.2, density=True)
        ax2.set_yticks([])   
        ax2.set_ylabel("")   

        ale_df = ale(X=X_set, model=model, feature=[f], grid_size=20, include_CI=True, plot=False)
        ax_ale.plot(ale_df.index, ale_df["eff"], marker="o", color="green") #ALE: Average local effect (correlation aware) 
        ax_ale.set_title(f"ALE", fontsize=8)
        ax_ale.set_xlabel(f, fontsize=6)
        ax_ale.set_ylabel("Effect (centered)", fontsize=6)
        ax_ale.tick_params(axis="both", labelsize=8)
        ax3 = ax_ale.twinx()
        ax3.hist(X_set[f], bins=30, color="gray", alpha=0.2, density=True)
        ax3.set_yticks([])
        ax3.set_ylabel("")

        shap.dependence_plot(f, shap_values, X_set, show=False, ax=ax_shap) #SHAP: Scatter of feature vs SHAP contributions (per sample) (interactions)
        ax_shap.set_title("SHAP Dependence", fontsize=8)
        ax_shap.set_xlabel(f, fontsize=6)
        ax_shap.set_ylabel("SHAP value", fontsize=6)
        ax_shap.tick_params(axis="both", labelsize=5)
        for cb in ax_shap.figure.get_axes():
            if cb is not ax_shap:
                cb.tick_params(labelsize=5)
                cb.set_ylabel(cb.get_ylabel(), fontsize=6)
        log_plot(fig, SHOW)
    
def plot_summary_shap(name, features, X, shap_values, plot_type="dot", SHOW=False):
    X_top = X[features]
    shap_values_top = shap_values[:, [X.columns.get_loc(f) for f in features]]
    fig = plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values_top, X_top, plot_type=plot_type, show=False) #SHAP summary: Importance + Direction
    plt.tight_layout()
    log_plot(fig, SHOW)

def plot_explanation_diag(name, X, shap_values, names, SHOW=False):
    fig = plt.figure(figsize=(12,6))
    if len(X)<=0:
        return
    elif len(X)==1:
        shap.plots.waterfall(shap_values[0], show=False)  #SHAP waterfall: Explains individual predictions
        plt.title(f"{names[0]}")
    else:
        shap.decision_plot(  #SHAP decision: Explains one or more predictions
            shap_values.base_values.mean(),   
            shap_values.values,
            X,
            legend_labels=names,
            legend_location="lower right",
            show=False
        )
    plt.tight_layout()
    log_plot(fig, SHOW)

class ModelWrapper:
    def __init__(self, name, build_fn, importance_fn, explainer_fn, params=None, metrics=None):
        self.name = name
        self.build_fn = build_fn
        self.importance_fn = importance_fn
        self.explainer_fn = explainer_fn
        self.params = params
        self.metrics = metrics
        self.datasets = None #[train_df, val_df, test_df, live_df]
        self.features = None
        self.model = None
        self.evals_result = None
        self.importances = None
        self.explainer = None
        self.X_eval = None
        self.y_eval = None
        self.y_pred = None
        self.top_features = None

    def get_training_data(self, datasets, features, target):
        results = []
        for df in dfs:
            X, y = df[features], df[target]
            results.append((X, y))
        self.datasets, self.features, self.target = datasets, features, target
        return results

    def fit(self, params, train_data, basepath, LOAD):
        if params is None: params = self.params
        assert self.build_fn is not None, f"{self.name} requires a build fn"
        assert params is not None, f"{self.name} requires params"
        self.params = params
        self.model, self.evals_result = self.build_fn(
            self.name, params,
            train_data,
            basepath=basepath, LOAD=LOAD
        )
        if self.importance_fn is not None:
            self.importances = self.importance_fn(self.model, self.features)
        if self.explainer_fn is not None:
            self.explainer = self.explainer_fn(self.model, X_background=train_data[0])
        self.initialise_eval()
        return self
    
    def get_model(self):
        return self.model
    def get_importances(self): 
        return self.importances
    def get_explainer(self):
        return self.explainer
    def get_datatsets(self):
        return self.datasets
    def get_features_target(self):
        return self.features, self.target
    
    def get_top_features(self, top_k, X_set=None):
        if X_set is None:
            X_set = self.X_eval      
        shap_values = self.explainer(X_set)    
        shap_df = get_global_SHAP(self.name, self.features, shap_values.values) 
        self.top_features = shap_df.head(top_k)["Feature"].to_list()
        return self.top_features
    
    def initialise_eval(self, X_eval=None, y_eval=None):
        if X_eval is None or y_eval is None:
            [[X_eval, y_eval]] = get_training_data([self.datasets[-2]], self.features, self.target)
        self.X_eval =  X_eval
        self.y_eval, self.y_pred = y_eval.to_numpy().ravel(), self.model.predict(X_eval) #np.exp(...)-1
        return self
    
    def results_summary(self, metrics=None):
        if metrics is None and self.metrics is None:
            metrics = ["r2", "rmse", "mae"]
        elif metrics is None:
            metrics = self.metrics
        self.metrics = metrics
        results_summary(self.y_eval, self.y_pred, metrics)
        return self
    
    def error_diag(self, metrics=None, SHOW=True):
        if metrics is None and self.metrics is None:
            metrics = ["rmse", "mae"]
        elif metrics is None:
            metrics = self.metrics
        self.metrics = metrics
        plot_learning_curve(self.name, self.evals_result, metrics, SHOW=SHOW) #new func hyperparameter tuning on validation set (validation curve and OOB with n estimators)
        plot_fitted_diag(self.name, self.y_eval, self.y_pred, SHOW=SHOW)
        plot_resid_diag(self.name, self.y_eval, self.y_pred, SHOW=SHOW)
        return self
    
    def importances_diag(self): #use SHAP instead
        get_importances(self.name, self.features, self.importances) #reduction in impurities (variance here), but features with high variance may appear more important
        get_permutation_importances(self.name, self.model, self.X_eval, self.y_eval) #increase in errors from shuffling a feature, but underestimates importance of correlated importances
        return self
    
    def SHAP_diag(self, top_k, X_set=None, SHOW=True):
        if X_set is None:
            X_set = self.X_eval      
        shap_values = self.explainer(X_set)  #Compares the models output with and without each feature relative to a baseline (doesn't use y)  
        shap_df = get_global_SHAP(self.name, self.features, shap_values.values) #SHAP: Contribution to prediction
        
        self.top_features = shap_df.head(top_k)["Feature"].to_list() 
        plot_summary_shap(self.name, self.top_features, X_set, shap_values.values, plot_type="violin", SHOW=SHOW)
        plot_dependence_diag(self.name, self.top_features, self.model, X_set, shap_values.values, SHOW=SHOW)
        return self
    
    def explanation_diag(self, symbols=None, live_df=None, SHOW=True):
        if live_df is None:
            live_df = self.datasets[-1]
        [[X_live, _]] = get_training_data([live_df], self.features, self.target)
        y_live = self.model.predict(X_live)

        shap_values = self.explainer(X_live) #explains individual predictions, y=baseline + sum(SHAP of features)
        get_live_pred(self.name, live_df["symbol"], y_live, shap_values.values, self.features)

        if symbols is None:
            return self
        if not isinstance(symbols, list):
            symbols = [symbols]
        selected_df = live_df[live_df["symbol"].isin(symbols)]
        X_selected = selected_df[self.features]
        shap_values = self.explainer(X_selected)
        plot_explanation_diag(self.name, X_selected, shap_values, selected_df["symbol"].to_list(), SHOW=SHOW)
        return self
    
    
    


    

