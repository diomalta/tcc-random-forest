# tcc-random-forest

## This script is executed by the container at startup.

> #!/bin/bash
> mysql -D abvdb -u root -p < /docker-entrypoint-initdb.d/database.sql

# Connect MySQL Database

> import mysql.connector
> Step 0: Connect to the database
> database = mysql.connector.connect(
> host="localhost",
> user="root",
> password="root",
> database="abvdb"
> )

> cursor = database.cursor()
> cursor.execute("SHOW TABLES")
> tables = cursor.fetchall()
> for x in tables:
> print(x)

# Cross validation

> https://medium.com/@edubrazrabello/cross-validation-avaliando-seu-modelo-de-machine-learning-1fb70df15b78
> from sklearn.model_selection import KFold, cross_val_score
> kf = KFold(n_splits=5, shuffle=True, random_state=123)
> cross_val_score(ml_pipe, train, y, cv=kf).mean() 0.813

# Step 1: Load the data

> NOTE: CHIKUNGUNYA = 0; DENGUE = 1; ZIKA = 2; OUTRAS_DOENCAS = 3
> LINK: https://earthly.dev/blog/plotting-rainfall-data-with-python-and-matplotlib/

# Step 2: Split the data into training and testing sets (30% for testing). random_state is used to ensure that the results are reproducible.

> https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
> Explain train_test_split: https://www.bitdegree.org/learn/train-test-split

# Step 6: Evaluate the model (metrics)

> print("Acurracy: ", accuracy_score(y_test, predictions))
> print("F1-Score: ", f1_score(y_test, predictions))
> print("Recall: ", recall_score(y_test, predictions))
> print("Precision: ", precision_score(y_test, predictions))

# Step 7: Save the model

> import pickle # https://www.geeksforgeeks.org/pickle-python/
> pickle.dump(randomForest, open("model.pkl", "wb"))

# Confusion Matrix

> https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62

# Understanding Random Forests Classifiers in Python Tutorial

> https://www.datacamp.com/tutorial/random-forests-classifier-python

# How to Develop a Random Forest Ensemble in Python

How to Develop a Random Forest Ensemble in Python

> https://machinelearningmastery.com/random-forest-ensemble-in-python/

# Colunas

> "NU_IDADE_N",
> "CS_SEXO",
> "CS_GESTANT",
> "CS_RACA",
> "CS_ZONA",
> "DIAS",
> "AUTO_IMUNE",
>
> "FEBRE",
> "MIALGIA",
> "CEFALEIA",
> "EXANTEMA",
> "VOMITO",
> "NAUSEA",
> "DOR_COSTAS",
> "CONJUNTVIT",
> "ARTRITE",
> "ARTRALGIA",
> "PETEQUIA_N",
> "LACO",
> "DOR_RETRO",
> "DIABETES",
> "HEMATOLOG",
> "HEPATOPAT",
> "RENAL",
> "HIPERTENSA",
> "ACIDO_PEPT",
>
> "CLASSI_FIN",

# Outra forma de recupera os dados de features/labels sem especificar

> X = dataset.iloc[:, 0:26].values
> y = dataset.iloc[:, 26].values

# Conjuntos de dados

## Dengue e chik

- https://data.mendeley.com/datasets/2d3kr8zynf/4

## Zika Virus

- https://www.openicpsr.org/openicpsr/project/115165/version/V2/view?path=/openicpsr/115165/fcr:versions/V2/SISA.csv&type=file#

# Matrix confusion 3x3

https://datapeaker.com/big-data/como-manejar-los-desafios-comunes-de-selenium-usando-python/
