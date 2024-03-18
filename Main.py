from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.evaluate import evaluate_output_model, print_evaluation_results
from src.init import X_train, X_test, y_train, y_test
from src.modelStacking import StackingEnsembleRegressor
from src.model import BaggingEnsembleRegressor

print("Primary energy consumption per capita (kWh/person)")
# param_grid = {

#     'base_estimator__n_estimators': [42, 100],
#     'n_estimators': [20, 30],
#     'max_samples': [0.8],
#     'max_features': [0.8]
# }
"""
#Bagging = BaggingEnsembleRegressor()
#Bagging.bayes_search_tune(X_train, y_train['Primary energy consumption per capita (kWh/person)'])
#Bagging.fit(X_train, y_train['Primary energy consumption per capita (kWh/person)'])
#predictions = Bagging.predict(X_test)
"""
stacking_ensemble = StackingEnsembleRegressor()

stacking_ensemble.fit(X_train, y_train)

predictions = stacking_ensemble.predict(X_test)

results = evaluate_output_model(predictions, y_test['Primary energy consumption per capita (kWh/person)'].to_numpy())

print_evaluation_results(results, model_name="Stacking Ensemble Regressor")


"""
avec Stacking

  MSE: 12602764.305371303
  MAE: 1998.2299787986278
  R2: 0.9895074379948218
"""

"""
avec Stacking et normalisation

  MSE: 0.0002124282610395725
  MAE: 0.007734014156865693
  R2: 0.9878053267852666
"""