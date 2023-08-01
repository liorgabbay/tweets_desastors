import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from preproccess_data import train_preprocess, preprocess_test


def KNN_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.DataFrame,
              loss_list: list):
    """
    Trains a K-Nearest Neighbors (KNN) classifier on the given training data, makes predictions on the validation data,
    and calculates the accuracy score.

    Parameters:
        X_train (pd.DataFrame): pandas DataFrame containing the training features.
        y_train (pd.Series): pandas Series containing the target labels for the training data.
        X_validation (pd.DataFrame): pandas DataFrame containing the validation features.
        y_validation (pd.DataFrame): pandas Series containing the target labels for the validation data.
        loss_list (list): A list to which the name of the model and its accuracy will be appended.

    Returns:
        None: The accuracy score of the KNN model will be appended to the loss_list.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_validation_scaled)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("Knn model", accuracy))


def decision_tree_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame,
                        y_validation: pd.DataFrame,
                        loss_list: list):
    """
    Trains a Decision Tree classifier on the given training data, makes predictions on the validation data,
    and calculates the accuracy score.

    Parameters:
        X_train (pd.DataFrame): pandas DataFrame containing the training features.
        y_train (pd.Series): pandas Series containing the target labels for the training data.
        X_validation (pd.DataFrame): pandas DataFrame containing the validation features.
        y_validation (pd.DataFrame): pandas Series containing the target labels for the validation data.
        loss_list (list): A list to which the name of the model and its accuracy will be appended.

    Returns:
        None: The accuracy score of the Decision Tree model will be appended to the loss_list.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train_scaled, y_train)
    y_pred = tree_model.predict(X_validation_scaled)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("decision tree model", accuracy))


def logistic_regression_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame,
                              y_validation: pd.DataFrame,
                              loss_list: list):
    """
    Trains a logistic regression classifier on the given training data using Decision Tree as the base estimator,
    makes predictions on the validation data, and calculates the accuracy score.

    Parameters:
        X_train (pd.DataFrame): pandas DataFrame containing the training features.
        y_train (pd.Series): pandas Series containing the target labels for the training data.
        X_validation (pd.DataFrame): pandas DataFrame containing the validation features.
        y_validation (pd.DataFrame): pandas Series containing the target labels for the validation data.
        loss_list (list): A list to which the name of the model and its accuracy will be appended.

    Returns:
        None: The accuracy score of the Logistic regression model will be appended to the loss_list.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)
    y_pred = logistic_model.predict(X_validation_scaled)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("logistic regression model", accuracy))


def adaboost_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame,
                   y_validation: pd.DataFrame,
                   loss_list: list):
    """
    Trains an AdaBoost classifier on the given training data using Decision Tree as the base estimator,
    makes predictions on the validation data, and calculates the accuracy score.

    Parameters:
        X_train (pd.DataFrame): pandas DataFrame containing the training features.
        y_train (pd.Series): pandas Series containing the target labels for the training data.
        X_validation (pd.DataFrame): pandas DataFrame containing the validation features.
        y_validation (pd.DataFrame): pandas Series containing the target labels for the validation data.
        loss_list (list): A list to which the name of the model and its accuracy will be appended.

    Returns:
        None: The accuracy score of the AdaBoost model will be appended to the loss_list.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    base_estimator = DecisionTreeClassifier(max_depth=1)
    adaboost_classifier = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)
    adaboost_classifier.fit(X_train_scaled, y_train)
    y_pred = adaboost_classifier.predict(X_validation_scaled)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("adaboost model", accuracy))


def SVM_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.DataFrame,
              loss_list: list):
    """
    Trains a Soft svm classifier on the given training data, makes predictions on the validation data,
    and calculates the accuracy score.

    Parameters:
        X_train (pd.DataFrame): pandas DataFrame containing the training features.
        y_train (pd.Series): pandas Series containing the target labels for the training data.
        X_validation (pd.DataFrame): pandas DataFrame containing the validation features.
        y_validation (pd.DataFrame): pandas Series containing the target labels for the validation data.
        loss_list (list): A list to which the name of the model and its accuracy will be appended.

    Returns:
        None: The accuracy score of the soft svm model will be appended to the loss_list.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    soft_svm = SVC(kernel='linear', C=1.0)
    soft_svm.fit(X_train_scaled, y_train)
    y_pred = soft_svm.predict(X_validation_scaled)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("soft svm model", accuracy))


def QDA_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.DataFrame,
              loss_list: list):
    """
        Trains a QDA_model classifier on the given training data, makes predictions on the validation data,
        and calculates the accuracy score.

        Parameters:
            X_train (pd.DataFrame): pandas DataFrame containing the training features.
            y_train (pd.Series): pandas Series containing the target labels for the training data.
            X_validation (pd.DataFrame): pandas DataFrame containing the validation features.
            y_validation (pd.DataFrame): pandas Series containing the target labels for the validation data.
            loss_list (list): A list to which the name of the model and its accuracy will be appended.

        Returns:
            None: The accuracy score of the QDA_model model will be appended to the loss_list.
        """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_validation)
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)
    y_pred = qda.predict(X_test_scaled)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("QDA model", accuracy))

def LDA_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.DataFrame,
              loss_list: list):
    """
        Trains a LDA classifier on the given training data, makes predictions on the validation data,
        and calculates the accuracy score.

        Parameters:
            X_train (pd.DataFrame): pandas DataFrame containing the training features.
            y_train (pd.Series): pandas Series containing the target labels for the training data.
            X_validation (pd.DataFrame): pandas DataFrame containing the validation features.
            y_validation (pd.DataFrame): pandas Series containing the target labels for the validation data.
            loss_list (list): A list to which the name of the model and its accuracy will be appended.

        Returns:
            None: The accuracy score of the LDA model will be appended to the loss_list.
        """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_validation)
    qda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.2)
    qda.fit(X_train_scaled, y_train)
    y_pred = qda.predict(X_test_scaled)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("LDA model", accuracy))
def neural_network_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame,
                         y_validation: pd.Series,
                         loss_list: list):
    """
    Trains a neural network classifier on the given training data,
    makes predictions on the validation data, and calculates the accuracy score.

    Parameters:
        X_train (pd.DataFrame): pandas DataFrame containing the training features.
        y_train (pd.Series): pandas Series containing the target labels for the training data.
        X_validation (pd.DataFrame): pandas DataFrame containing the validation features.
        y_validation (pd.Series): pandas Series containing the target labels for the validation data.
        loss_list (list): A list to which the name of the model and its accuracy will be appended.

    Returns:
        None: The accuracy score of the neural network model will be appended to the loss_list.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    neural_net = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
    neural_net.fit(X_train_scaled, y_train)
    y_pred = neural_net.predict(X_validation_scaled)
    accuracy = accuracy_score(y_validation, y_pred)
    loss_list.append(("neural network model", accuracy))


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
    logistic_regression_model(X_train, y_train, X_validation, y_validation, model_accuracy)
    QDA_model(X_train, y_train, X_validation, y_validation, model_accuracy)
    LDA_model(X_train, y_train, X_validation, y_validation, model_accuracy)
    # SVM_model(X_train, y_train, X_validation, y_validation, model_accuracy)
    neural_network_model(X_train, y_train, X_validation, y_validation, model_accuracy)
    for model in model_accuracy:
        print(model[0], f"accuracy: {model[1]}")
