import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def avg(data):
    sum = 0
    for val in data:
        sum+=val
    return sum/len(data)


def main():
    data = pd.read_csv("data_C02_emission.csv")
    car_emission = pd.DataFrame(data)

    car_emission.drop_duplicates(inplace=True)
    print(car_emission)
    print(car_emission.shape[0])

    car_emission.info()
    print("3 lowest: ", car_emission.nsmallest(3, "Fuel Consumption City (L/100km)")[["Make", "Model", "Fuel Consumption City (L/100km)"]])
    print("3 largest: ", car_emission.nlargest(3, "Fuel Consumption City (L/100km)")[["Make", "Model", "Fuel Consumption City (L/100km)"]])
    
    engine_interval = car_emission["Engine Size (L)"].between(2.5, 3.5)
    only_25 = car_emission[(car_emission["Engine Size (L)"] == 2.5) & (car_emission["Engine Size (L)"] == 3.5)]
    print(engine_interval)

    filtered_by_engine = car_emission[engine_interval]

    avg_emission = avg(filtered_by_engine["CO2 Emissions (g/km)"])
    print("Avg emission for engine 2.5-3.5", np.round(avg_emission, 2))

    all_audi = car_emission[car_emission["Make"] == "Audi"]
    print(all_audi.shape[0])
    all_audi_four_cylinder = all_audi[all_audi["Cylinders"] == 4]

    print("Avg all audi 4 cylinder emission: ", avg(all_audi_four_cylinder["CO2 Emissions (g/km)"]))

    all_four_cylinder = car_emission[car_emission["Cylinders"] == 4]
    all_six_cylinder = car_emission[car_emission["Cylinders"] == 6]
    all_eight_cylinder = car_emission[car_emission["Cylinders"] == 8]
    all_twelve_cylinder = car_emission[car_emission["Cylinders"] == 12]

    print("Avg emission 4 cylinder: ", avg(all_four_cylinder["CO2 Emissions (g/km)"]))
    print("Avg emission 6 cylinder: ", avg(all_six_cylinder["CO2 Emissions (g/km)"]))
    print("Avg emission 8 cylinder: ", avg(all_eight_cylinder["CO2 Emissions (g/km)"]))
    print("Avg emission 12 cylinder: ", avg(all_twelve_cylinder["CO2 Emissions (g/km)"]))

    all_regular_gasoline = car_emission[car_emission["Fuel Type"] == "X"]
    all_diesel = car_emission[car_emission["Fuel Type"] == "D"]

    print("Avg emission regular gas: ", avg(all_regular_gasoline["Fuel Consumption City (L/100km)"]))
    print("Avg emission diesel: ", avg(all_diesel["Fuel Consumption City (L/100km)"]))
    print("Med emission regular gas: ", np.median(all_regular_gasoline["Fuel Consumption City (L/100km)"]))
    print("Med emission diesel: ", np.median(all_diesel["Fuel Consumption City (L/100km)"]))

    four_cylinder_diesel = all_four_cylinder[all_four_cylinder["Fuel Type"]=="D"]
    max_four_cylinder_diesel = four_cylinder_diesel[four_cylinder_diesel["Fuel Consumption City (L/100km)"] == np.max(four_cylinder_diesel["Fuel Consumption City (L/100km)"])]
    print("Four cylinder diesel max consumption: ", max_four_cylinder_diesel["Model"])

    all_manuals = car_emission[car_emission["Transmission"] == "M"]
    print("Manuals count: ", all_manuals.shape)

    print("Correlation: ", car_emission.corr( numeric_only = True ))


if __name__=="__main__":
    main()