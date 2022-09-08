import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load the data
dataset = pd.read_csv("data/dataset-denv-chyk-only.csv", delimiter= ';')

# Step 2: Split the data into training and testing sets (30% for testing). random_state is used to ensure that the results are reproducible.
X = dataset.iloc[:, 0:26].values
y = dataset.iloc[:, 26].values

# "NU_IDADE_N", "CS_SEXO", "CS_GESTANT", "CS_RACA", "CS_ZONA", "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA", "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N", "LACO", "DOR_RETRO", "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL", "HIPERTENSA", "ACIDO_PEPT", "AUTO_IMUNE", "DIAS", "CLASSI_FIN
# X=dataset[["MIALGIA", "CEFALEIA", "EXANTEMA", "NAUSEA", "ARTRITE", "DOR_RETRO"]]  # Features
# y=dataset['CLASSI_FIN']  # Labels

from sklearn.model_selection import KFold, cross_val_score

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
kf = KFold(n_splits=2, shuffle=True, random_state=123)
# print(cross_val_score(randomForest, X, y, cv=kf))
# # .mean()

from sklearn.model_selection import GridSearchCV

randomForest = RandomForestClassifier()

param_grid = {
  'n_estimators': [100, 200, 300, 400, 500],
  'max_depth': [2, 5, 7, 10, 15, 20, 25, 30],
  'class_weight': ['balanced', 'balanced_subsample', None],
  'max_features': ['sqrt', 'log2', None],
  'random_state': [24, 42, 100, 123, 200]
}

grid_clf = GridSearchCV(randomForest, param_grid, cv=10)
grid_clf.fit(X_train, y_train)

# print(grid_clf.best_estimator_)
# print()
# print(grid_clf.best_score_)
# print()
print(grid_clf.best_params_)
# print()
# print(grid_clf.cv_results_)


# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     # Step 3: Create a model and train it
#     randomForest = RandomForestClassifier()
#     randomForest.fit(X_train, y_train)

#     y_pred = randomForest.predict(X_test)

#     # Step 6: Evaluate the model (metrics)
#     tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

#     # Interpretação: Você previu positivo e é verdade.
#     print("True Positives: ", tp)
#     # Interpretação: Você previu negativo e é verdade.
#     print("True Negatives: ", tn)
#     # Interpretação: Você previu positivo e é falso.
#     print("False Positives: ", fp)
#     # Interpretação: Você previu negativo e é falso.
#     print("False Negatives: ", fn)
#     print()
#     # print(classification_report(y_test,y_pred))
#     # print("#################################")
#     print("Acurracy: ", accuracy_score(y_test, y_pred))
#     print()

# # Step 4: Train the model
# randomForest = RandomForestClassifier(n_estimators=500, n_jobs=2)

# randomForest.fit(X_train, y_train)

# # Step 5: Predict the model
# y_pred = randomForest.predict(X_test)

# # Step 6: Evaluate the model (metrics)
# tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

# # Interpretação: Você previu positivo e é verdade.
# print("True Positives: ", tp)
# # Interpretação: Você previu negativo e é verdade.
# print("True Negatives: ", tn)
# # Interpretação: Você previu positivo e é falso.
# print("False Positives: ", fp)
# # Interpretação: Você previu negativo e é falso.
# print("False Negatives: ", fn)
# print()
# # print(classification_report(y_test,y_pred))
# # print("#################################")
# print("Acurracy: ", accuracy_score(y_test, y_pred))

# feature_imp = pd.Series(randomForest.feature_importances_,index=dataset.columns[: 26]).sort_values(ascending=False)

# print()
# print(feature_imp)

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Creating a bar plot
# sns.barplot(x=feature_imp, y=feature_imp.index)

# # Add labels to your graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.show()