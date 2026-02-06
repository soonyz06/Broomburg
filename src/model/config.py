from src.model.results import ModelWrapper
from src.model.initialise.rf import build_rf, rf_importances
from src.model.initialise.xgb import build_xgb, xgb_importances
import shap

def tree_explainer(model, X_background):
    return shap.TreeExplainer(model, data=X_background)

def make_rf():
    name = "RF1"
    metrics = None
    params = {
        "n_estimators": 100,
        "random_state": 42
    }
    return ModelWrapper(name, build_rf, rf_importances, tree_explainer, params, metrics)

def make_xgb():
    name = "XGB1"
    objectives = ["reg:squarederror", "reg:quantileerror", "reg:pseudohubererror"]
    metrics = ["rmse", "mae", "mape"]
    params = {
        "objective": objectives[0],
        "reg_alpha": 0.5,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": metrics,
        "random_state": 42,
        "n_estimators": 100
    }
    return ModelWrapper(name, build_xgb, xgb_importances, tree_explainer, params, metrics)
