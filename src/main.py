# Import dependencies
import mysql.connector

# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import RandomForestClassifier
# from pandas import DataFrame
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_fscore_support

# Step 0: Connect to the database
database = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="abvdb"
)

cursor = database.cursor()
cursor.execute("SHOW TABLES")
tables = cursor.fetchall()
for x in tables:
  print(x)

# Step 1: Load the data
# cancer = load_breast_cancer()
# print(cancer.target_names)
# print(cancer.feature_names)

# Step 2: Split the data into training and testing sets (30% for testing). random_state is used to ensure that the results are reproducible.
# X, y = load_breast_cancer(return_X_y = True)
# print(cancer.target_names)
# print(cancer.target)

# Explain train_test_split: https://www.bitdegree.org/learn/train-test-split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

# Step 3 (Optional): Create a DataFrame of the data to show rows and columns
# dataset = DataFrame(X_train, columns = cancer.feature_names)
# print(dataset.head())

# Step 4: Train the model
# randomForest = RandomForestClassifier(n_estimators = 100)
# randomForest.fit(X_train, y_train)

# Step 5: Predict the model
# predictions = randomForest.predict(X_test)
# print(predictions)

# Step 6: Evaluate the model (metrics)
# print("Acurracy: ", accuracy_score(y_test, predictions))
# print("F1-Score: ", f1_score(y_test, predictions))
# print("Recall: ", recall_score(y_test, predictions))
# print("Precision: ", precision_score(y_test, predictions))

# Step 7: Save the model
# import pickle # https://www.geeksforgeeks.org/pickle-python/
# pickle.dump(randomForest, open("model.pkl", "wb"))
