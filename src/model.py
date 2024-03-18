from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd
import os


class BaggingEnsembleRegressor:
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=0.8, max_features=0.8):
        if base_estimator is None:
            base_estimator = CatBoostRegressor(iterations=100, random_seed=42, verbose=False)
        self.base_estimator = base_estimator
        self.bagging_regressor = BaggingRegressor(
            base_estimator=self.base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=42
        )

    def bayes_search_tune(self, X_train, y_train):
        search_spaces = {
            'base_estimator__iterations': Integer(50, 200),
            'base_estimator__depth': Integer(1, 8),
            'n_estimators': Integer(10, 20),
            'max_samples': Real(0.5, 1.0),
            'max_features': Real(0.5, 1.0)
        }

        bayes_search = BayesSearchCV(estimator=self.bagging_regressor,
                                     search_spaces=search_spaces,
                                     n_iter=32, cv=5, scoring='neg_mean_absolute_error',
                                     n_jobs=-1, verbose=3, random_state=42)
        bayes_search.fit(X_train, y_train)
        self.bagging_regressor = bayes_search.best_estimator_
        print("Meilleurs paramètres trouvés:", bayes_search.best_params_)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    def fit(self, X_train, y_train):
        print("Training Bagging Ensemble Regressor...")
        self.bagging_regressor.fit(X_train, y_train)

    def predict(self, X_test):
        print("Predicting with Bagging Ensemble Regressor...")
        return self.bagging_regressor.predict(X_test)

    def save_results(self, metrics, filepath='model_results.csv'):
        data = {
            'Modèle': 'Bagging CatBoostRegressor',
            'Assemblage': f"{self.bagging_regressor.n_estimators} estimators, {self.bagging_regressor.max_samples} max_samples, {self.bagging_regressor.max_features} max_features",
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'R2': metrics['R2']
        }
        df_new = pd.DataFrame([data])

        if os.path.exists(filepath):
            df_old = pd.read_csv(filepath)
            df_result = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_result = df_new

        df_result.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")