import utils
import time

from sklearn.model_selection import train_test_split

# Passo 1: Carregar os dados
X, y = utils.get_dataset()

# Passo 2: Separar os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# Passo 3: Treine o modelo
model = utils.get_random_forest_model()
start_time = time.time()
model.fit(X_train, y_train.values.ravel())
# print("Treinamento --- %s seconds ---" % (time.time() - start_time))

# Passo 4: Preveja o modelo
start_time2 = time.time()
y_pred = model.predict(X_test)
# print("Teste --- %s seconds ---" % (time.time() - start_time2))

# Passo 5: Avalie o modelo (métricas)
utils.show_classification_report(y_test, y_pred)
utils.show_confusion_matrix(y_test, y_pred)

# NOTA: features mais importante para o modelo
utils.show_feature_importances(model)
