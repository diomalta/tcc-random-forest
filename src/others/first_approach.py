import utils
import time

from sklearn.model_selection import train_test_split

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

# Passo 1: Carregar os dados
X, y = utils.get_dataset()

# Passo 2: Separar os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# Passo 3: Treine o modelo
model = utils.get_random_forest_model()
start_time = time.time()
model.fit(X_train, y_train.values.ravel())
print("Treinamento --- %s seconds ---" % (time.time() - start_time))

# Passo 4: Preveja o modelo
start_time2 = time.time()
y_pred = model.predict(X_test)
print("Teste --- %s seconds ---" % (time.time() - start_time2))

# Passo 5: Avalie o modelo (métricas)
utils.show_classification_report(y_test, y_pred)
# utils.show_confusion_matrix(y_test, y_pred)

# NOTA: features mais importante para o modelo
# utils.show_feature_importances(model)

# save model classificator  
# import os
# import joblib

# # joblib.dump(model, "./random_forest.joblib")
# loaded_rf = joblib.load("./random_forest.joblib")

# # https://www.youtube.com/watch?v=XgwxZQZjP38
# print(loaded_rf.predict(X))
# print(loaded_rf.predict_proba(X))

# add classification_report
# from sklearn.metrics import classification_report
# print(classification_report(y, loaded_rf.predict_proba(X)))
# predict_proba = loaded_rf.predict_proba(X)