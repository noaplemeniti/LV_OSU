from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error

def main():
    data = pd.read_csv("data_C02_emission.csv")
    print(data.shape)

    scaler = MinMaxScaler()

    X = data[
    [
        "Fuel Consumption City (L/100km)",
        "Fuel Consumption Hwy (L/100km)",
        "Fuel Consumption Comb (L/100km)",
        "Engine Size (L)",
        "Cylinders"
    ]]

    Y = data[["CO2 Emissions (g/km)"]]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2)

    plt.scatter(X_TRAIN["Fuel Consumption City (L/100km)"], Y_TRAIN, color="blue")
    plt.scatter(X_TEST["Fuel Consumption City (L/100km)"], Y_TEST, color="red")
    plt.show()

    #standardizacija
    X_TRAIN_N = scaler.fit_transform(X_TRAIN)
    X_TEST_N = scaler.transform(X_TEST)

    print(X_TRAIN)

    plt.hist(X_TRAIN)
    plt.show()
    plt.hist(X_TRAIN_N)
    plt.show()

    linearModel = lm.LinearRegression()
    linearModel.fit(X_TRAIN_N, Y_TRAIN)
    print(linearModel.coef_)

    Y_PREDICT = linearModel.predict(X_TEST_N)
    MAE = mean_absolute_error(Y_TEST, Y_PREDICT)
    print(MAE)

    plt.scatter(X_TEST_N.T[0], Y_TEST, color="blue")
    plt.scatter(X_TEST_N.T[0], Y_PREDICT, color="red")
    plt.show()


if __name__=="__main__":
    main()
