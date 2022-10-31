import utils
import pandas

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

FEATURES_V2 = [
  "FEBRE", 
  "MIALGIA", 
  "CEFALEIA", 
  "EXANTEMA", 
  "VOMITO", "NAUSEA", 
  "DOR_COSTAS", 
  "ARTRITE", 
  "ARTRALGIA", 
  "PETEQUIA_N", 
  "DOR_RETRO"
  ]

X, y = utils.get_dataset(name="dataset-denv-chyk.csv", features=FEATURES_V2)
model = utils.get_random_forest_model()

k_folds = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(model, X, y.values.ravel(), cv=k_folds)

print("Média (Desvio Padrão): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "\n")

list_result = pandas.Series(scores).sort_values(ascending=False)
print("Tabela (DESC):", "\n", list_result)

# Creating a bar plot

import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x=list_result.index, y=list_result)

# Add labels to your graph
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.title("KFold 10")
plt.show()


# NOTE: Processo de analise se existe variação nos resultados com base na alteração das amostras
#       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

# # 11.449 / 10 = 1.144,9 (K-Fold com 10 folds)
# print("Média (Desvio Padrão): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "\n")
