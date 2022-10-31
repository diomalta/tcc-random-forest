# DEFAULT_FEATURES = ["FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA", "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N", "LACO", "DOR_RETRO", "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL", "HIPERTENSA", "ACIDO_PEPT"]
DEFAULT_FEATURES = ["FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA", "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N", "LACO", "DOR_RETRO"]

def get_dataset(name = "dataset-denv-chyk.csv", features = DEFAULT_FEATURES, labels = ["CLASSI_FIN"]):
    import pandas as pd
    
    dataset = pd.read_csv("data/" + name, delimiter= ';')

    X=dataset[features]  # Features
    y=dataset[labels]  # Labels

    return X, y

def get_random_forest_model(
        n_estimators=200,
        n_jobs=5,
        max_features="sqrt",
    ):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
            n_estimators=n_estimators, 
            n_jobs=n_jobs,
            max_features=max_features
        )

def show_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred)
    # tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

    # print("True Positives: ", tp)
    # # Interpretação: Você previu negativo e é verdade.
    # print("True Negatives: ", tn)
    # # Interpretação: Você previu positivo e é falso.
    # print("False Positives: ", fp)
    # # Interpretação: Você previu negativo e é falso.
    # print("False Negatives: ", fn)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Rótulos previstos');ax.set_ylabel('Rótulos verdadeiros')
    ax.set_title('Matriz de confusão');
    ax.xaxis.set_ticklabels(['0', '1', '2']); ax.yaxis.set_ticklabels(['0', '1', '2'])

    plt.show()

def show_classification_report(y_test, y_pred):
    from sklearn.metrics import classification_report
    print("Relatório de Classificação", "\n")
    print(classification_report(y_test, y_pred))

def show_feature_importances(model, features = DEFAULT_FEATURES):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    importances = pd.Series(model.feature_importances_,index=features).sort_values(ascending=False)
    print(importances)

    # Creating a bar plot
    sns.barplot(x=importances, y=importances.index)

    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.show()

def get_k_folds(ns = 5):
    from sklearn.model_selection import KFold
    return KFold(n_splits=ns, shuffle=True, random_state=123)

def grid_search(model, X, y, parameters, cv):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, accuracy_score

    grid_obj = GridSearchCV(model, parameters, scoring=make_scorer(accuracy_score), cv=cv)
    grid_fit = grid_obj.fit(X, y.values.ravel())

    return grid_fit.best_estimator_, grid_fit.best_score_, grid_fit.best_params_, grid_fit.cv_results_