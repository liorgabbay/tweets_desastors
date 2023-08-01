import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from preproccess_data import train_preprocess, preprocess_test


def KNN_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.DataFrame,
              loss_list: list):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_validation)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("Knn model", accuracy))


def decision_tree_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame,
                        y_validation: pd.DataFrame,
                        loss_list: list):
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_validation)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("decision tree model", accuracy))


def adaboost_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame,
                   y_validation: pd.DataFrame,
                   loss_list: list):
    base_estimator = DecisionTreeClassifier(max_depth=1)
    adaboost_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=42)
    adaboost_classifier.fit(X_train, y_train)
    y_pred = adaboost_classifier.predict(X_validation)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("adaboost model", accuracy))


if __name__ == '__main__':
    train_data = pd.read_csv("./data/S_train_data.csv")
    validation_data = pd.read_csv("./data/validation_data.csv")
    y_train = train_data['target']
    X_train = train_data.drop(columns=['target'])
    y_validation = validation_data['target']
    X_validation = validation_data.drop(columns=['target'])
    X_train, y_train, avg = train_preprocess(X_train, y_train)
    X_validation = preprocess_test(X_validation, avg)
    model_accuracy = []

    KNN_model(X_train, y_train, X_validation, y_validation, model_accuracy)
    decision_tree_model(X_train, y_train, X_validation, y_validation, model_accuracy)
    adaboost_model(X_train, y_train, X_validation, y_validation, model_accuracy)
    for model in model_accuracy:
        print(model[0], f"accuracy: {model[1]}")
