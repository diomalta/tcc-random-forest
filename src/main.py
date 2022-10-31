import utils

# Step 1: Load the data
X, y = utils.get_dataset()
model = utils.get_random_forest_model()

param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [2, 5, 7, 10],
  'class_weight': ['balanced', 'balanced_subsample'],
  'max_features': ['sqrt', 'log2'],
  'random_state': [24, 42, 123]
}
best_estimator, best_score, best_params, cv_results = utils.grid_search(model, X, y, param_grid, utils.get_k_folds(10))
print(best_estimator)
print()
print(best_score)
print()
print(best_params)
# print()
# print(cv_results)
