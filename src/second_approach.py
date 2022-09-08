import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load the data
dataset = pd.read_csv("data/dataset-denv-chyk-only.csv", delimiter= ';')

# Step 2: Split the data into training and testing sets (30% for testing). random_state is used to ensure that the results are reproducible.
X=dataset[["FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA", "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N", "LACO", "DOR_RETRO", "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL", "HIPERTENSA", "ACIDO_PEPT",]]  # Features
y=dataset['CLASSI_FIN']  # Labels

# X = dataset.iloc[:, 0:26].values
# y = dataset.iloc[:, 26].values

from sklearn.model_selection import KFold, cross_val_score

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

kf = KFold(n_splits=10, shuffle=True, random_state=123)

# randomForest = RandomForestClassifier()
# print(cross_val_score(randomForest, X, y, cv=kf))
# print(cross_val_score(randomForest, X, y, cv=kf).mean()) # is used to calculate the average of the scores

# Split data to get best model random forest with 10 fold cross validation
models = dict()
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    randomForest = RandomForestClassifier()
    randomForest.fit(X_train, y_train)
    y_pred = randomForest.predict(X_test)

    print("Relatório de Classificação", "\n")
    print(classification_report(y_test, y_pred))
