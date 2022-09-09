import utils
import pandas

from sklearn.model_selection import cross_val_score

# Step 1: Load the data
X, y = utils.get_dataset()
model = utils.get_random_forest_model()

# NOTE: Processo de analise se existe variação nos resultados com base na alteração das amostras
#       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
k_folds = utils.get_k_folds(10)
scores = cross_val_score(model, X, y.values.ravel(), cv=k_folds)

# 11.449 / 10 = 1.144,9 (K-Fold com 10 folds)
print("Média (Desvio Padrão): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "\n")

list_result = pandas.Series(scores).sort_values(ascending=False)
print("Tabela (DESC):", "\n", list_result)
