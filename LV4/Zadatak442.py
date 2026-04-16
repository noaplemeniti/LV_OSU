from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn . preprocessing import OneHotEncoder
from sklearn.metrics import max_error

def main():
    data = pd.read_csv("data_C02_emission.csv")
    print(data.shape)

    scaler = MinMaxScaler()

    ohe = OneHotEncoder ()

    X = data[
    [
        "Fuel Consumption City (L/100km)",
        "Fuel Consumption Hwy (L/100km)",
        "Fuel Consumption Comb (L/100km)",
        "Engine Size (L)",
        "Cylinders"
    ]]

    X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()

    X = np.array(X)
    X_encoded = np.array(X_encoded)

    X = np.column_stack((X, X_encoded))

    print(X)

    Y = data[["CO2 Emissions (g/km)"]]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2)

    linearModel = lm.LinearRegression()
    linearModel.fit(X_TRAIN, Y_TRAIN)
    Y_PREDICT = linearModel.predict(X_TEST)

    max_error_ = max_error(Y_TEST, Y_PREDICT)
    print(max_error_)

    plt.scatter(X_TEST.T[0], Y_TEST, color="blue")
    plt.scatter(X_TEST.T[0], Y_PREDICT, color="red")
    plt.show()

if __name__=="__main__":
    main()