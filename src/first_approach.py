import pandas as pd
import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#####################################################################
# DESCREVER A PRIMEIRA ETAPA DO DESENVOLVIMENTO
# Em relação ao dataset, o que você fez?
# Em relação ao modelo, o que você fez?
# Em relação ao treinamento, o que você fez?
# Em relação à avaliação, o que você fez?
# Em relação à features, o que você fez?
# Abordar os pontos que devem ser melhorados nas proximas abordagens
#####################################################################

# Step 1: Load the data
X, y = utils.get_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# Step 4: Train the model
model = utils.get_random_forest_model()
model.fit(X_train, y_train.values.ravel())

# Step 5: Predict the model
y_pred = model.predict(X_test)

# Step 6: Evaluate the model (metrics)
utils.show_classification_report(y_test, y_pred)

utils.show_confusion_matrix(y_test, y_pred)
utils.show_feature_importances(model)
