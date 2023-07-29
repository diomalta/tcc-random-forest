import utils

FEATURES_V2 = [
  "FEBRE", 
  "MIALGIA", 
  "CEFALEIA", 
  "EXANTEMA", 
  "VOMITO", 
  "NAUSEA", 
  "DOR_COSTAS", 
  "ARTRITE", 
  "ARTRALGIA", 
  "PETEQUIA_N", 
  "DOR_RETRO"
]

# Step 1: Load the data
X, y = utils.get_dataset(name="dataset-denv-chyk-only.csv", features=FEATURES_V2)
model = utils.get_random_forest_model()

param_grid = {
  'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
  'max_depth': [2, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50],
  'class_weight': ['balanced', 'balanced_subsample'],
  'max_features': ['sqrt', 'log2'],
  'random_state': [24, 42, 123, 666, 999],
  'n_jobs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
  'criterion': ['gini', 'entropy'],
}

best_estimator, best_score, best_params, cv_results = utils.grid_search(model, X, y, param_grid, utils.get_k_folds(10))

print("best_estimator")
print(best_estimator)
print("best_score")
print(best_score)
print("best_params")
print(best_params)
# print("cv_results")
# print(cv_results)
