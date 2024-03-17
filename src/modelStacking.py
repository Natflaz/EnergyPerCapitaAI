from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


class StackingEnsembleRegressor:
    def __init__(self):
        self.models = [
            ('lightgbm', LGBMRegressor(random_state=42)),
            ('catboost', CatBoostRegressor(random_state=42)),
        ]
        self.stackingRegressor = None

    def optimize_lightgbm(self, X_train, y_train):
        lgbm_search_space = {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(1, 8),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'colsample_bytree': Real(0.5, 1.0, prior='uniform')
        }
        lgbm_model = LGBMRegressor(random_state=42)
        lgbm_search = BayesSearchCV(lgbm_model, lgbm_search_space, n_iter=32, cv=5,
                                    scoring='neg_mean_absolute_error', n_jobs=-1, verbose=3, random_state=42)
        lgbm_search.fit(X_train, y_train)
        return lgbm_search.best_estimator_

    def optimize_catboost(self, X_train, y_train):
        catboost_search_space = {
            'iterations': Integer(50, 200),
            'depth': Integer(1, 8),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.5, 1.0, prior='uniform')
        }
        catboost_model = CatBoostRegressor(random_state=42, verbose=0)
        catboost_search = BayesSearchCV(catboost_model, catboost_search_space, n_iter=32, cv=5,
                                        scoring='neg_mean_absolute_error', n_jobs=-1, verbose=3, random_state=42)
        catboost_search.fit(X_train, y_train)
        return catboost_search.best_estimator_

    def create_stacking_regressor(self, X_train, y_train):
        optimized_lightgbm = self.optimize_lightgbm(X_train, y_train)
        optimized_catboost = self.optimize_catboost(X_train, y_train)
        self.models = [
            ('lightgbm', optimized_lightgbm),
            ('catboost', optimized_catboost),
        ]
        self.stackingRegressor = StackingRegressor(
            estimators=self.models,
            final_estimator=XGBRegressor(n_estimators=100)
        )

    def fit(self, X_train, y_train):
        if not self.stackingRegressor:
            self.create_stacking_regressor(X_train, y_train)
        print("Training Stacking Ensemble Regressor...")
        self.stackingRegressor.fit(X_train, y_train)

    def predict(self, X_test):
        print("Predicting with Stacking Ensemble Regressor...")
        return self.stackingRegressor.predict(X_test)