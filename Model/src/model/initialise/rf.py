import joblib
from sklearn.ensemble import RandomForestRegressor

def build_rf(name, params, train_data, basepath=None, LOAD=False):
    if LOAD:
        print(f"\n[INFO]Loading {name} Model")
        if basepath is None:
            return None
        rf = joblib.load(basepath / f"{name}_model.pkl")
    else:
        print(f"\n[INFO]Training {name} Model")
        X_train, y_train = train_data[:2]
        rf = RandomForestRegressor( #regime~Quantile RF?
            **params
        )
        rf.fit(X_train, y_train.to_numpy().ravel())
        if basepath is not None:
            joblib.dump(rf, basepath / f"{name}_model.pkl")
    return rf, None
    
def rf_importances(model, *args):
    return  model.feature_importances_
