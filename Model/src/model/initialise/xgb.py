import json
import numpy as np
from xgboost import XGBRegressor

def build_xgb(name, params, train_data, basepath=None, LOAD=False):
    if LOAD:
        print(f"\n[INFO]Loading {name} Model")
        if basepath is None:
            return None
        bst = XGBRegressor()
        bst.load_model(basepath / f"{name}_model.json")
        with open(basepath / f"{name}_evals.json", "r") as f:
            evals_result = json.load(f)
    else:
        print(f"\n[INFO]Training {name} Model")
        X_train, y_train, X_val, y_val = train_data[:4]
        bst = XGBRegressor(
            **params
        )
        bst.fit(
            X_train, y_train,
            eval_set = [(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        evals_result = bst.evals_result()
        evals_result["train"] = evals_result.pop("validation_0")
        evals_result["val"] = evals_result.pop("validation_1")
        
        if basepath is not None:
            bst.save_model(basepath / f"{name}_model.json")
            with open(basepath / f"{name}_evals.json", "w") as f:
                json.dump(evals_result, f)
    return bst, evals_result

def xgb_importances(model, features, importance_type="gain"):              
    scores = model.get_booster().get_score(importance_type=importance_type)
    return np.array([scores.get(f, 0.0) for f in features])
#weight: counts how many times a feature is picked
#gain: measures average improvement in accuracy
#cover: reflects the number of samples affcted by the feature
