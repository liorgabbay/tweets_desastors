import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from preproccess_data import train_preprocess,preprocess_test
from models import logistic_regression_model

def run_model(X_train:pd.DataFrame,y_train:pd.Series,X_test:pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)
    y_pred = logistic_model.predict(X_test_scaled)
    return y_pred



if __name__ == '__main__':
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    y_train = train_data["target"]
    X_train = train_data.drop(columns=["target"])
    X_train,y_train,avg =train_preprocess(X_train,y_train)
    X_test = preprocess_test(test_data,avg)
    y_pred = run_model(X_train,y_train,X_test)
    result = pd.concat([test_data,pd.Series(y_pred,name="target")],axis=1)
    result.to_csv("./data/result.csv",index= False)


