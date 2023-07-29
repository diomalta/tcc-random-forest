import utils
import pandas

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

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

X, y = utils.get_dataset(name="dataset-denv-chyk-only.csv", features=FEATURES_V2)
model = utils.get_random_forest_model()

k_folds = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(model, X, y.values.ravel(), cv=k_folds)

print("Média (Desvio Padrão): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "\n")

list_result = pandas.Series(scores).sort_values(ascending=False)
# floor values from array
list_result = list_result.apply(lambda x: round(x, 2))
print("Tabela (DESC):", "\n", list_result)

# Passo 5: Avalie o modelo (métricas)
utils.show_classification_report(y_test, y_pred)
utils.show_confusion_matrix(y_test, y_pred)

# Creating a bar plot

import matplotlib.pyplot as plt
import seaborn as sns

# sns.barplot(x=list_result, y=list_result.index)
sns.barplot(x=list_result.index, y=list_result)

# Add labels to your graph
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.title("Resultados dos 10 subconjuntos")
plt.show()


# NOTE: Processo de analise se existe variação nos resultados com base na alteração das amostras
#       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

# # 11.449 / 10 = 1.144,9 (K-Fold com 10 folds)
# print("Média (Desvio Padrão): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "\n")
