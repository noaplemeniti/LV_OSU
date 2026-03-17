import matplotlib.pyplot as plt
import pandas as pd

def main():
    data = pd.read_csv("data_C02_emission.csv")

    plt.hist(data["CO2 Emissions (g/km)"])
    plt.show()

    co2_emission = data["CO2 Emissions (g/km)"]
    city_fuel = data["Fuel Consumption City (L/100km)"]

    plt.scatter(city_fuel, co2_emission)
    plt.show()

    data.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")
    plt.show()

    fuel_counts = data.groupby("Fuel Type").size()
    fuel_counts.plot(kind="bar")
    plt.show()

    co2_by_cyl = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
    co2_by_cyl.plot(kind="bar")
    plt.show()

if __name__=="__main__":
    main()