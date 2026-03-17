import numpy as np
import matplotlib.pyplot as plt

def avg(data):
    total = 0
    for _ in data:
        total += _
    return total/len(data)

def output_info(path):
    data = np.genfromtxt(path, delimiter=',', names=True)
    print(data.shape)
    height = data["Height"]
    weight = data["Weight"]
    plt.scatter(height, weight)
    height50 = height[30:40:2]
    weight50 = weight[30:40:2]

    _men_only = (data["Gender"] == 1)
    men_only_height = data["Height"][_men_only]
    men_only_weight = data["Weight"][_men_only]
    plt.scatter(men_only_height, men_only_weight)

    #plt.scatter(height50, weight50)
    plt.show()
    height_avg = avg(height)
    height_max = max(height)
    height_min = min(height)
    print(height_avg, height_max, height_min)
    men_only = (data["Gender"] == 1)
    women_only = (data["Gender"] == 0)
    men_height = data["Height"][men_only]
    women_height = data["Height"][women_only]
    women_height_avg = avg(women_height)
    women_height_max = max(women_height)
    women_height_min = min(women_height)
    men_height_avg = avg(men_height)
    men_height_max = max(men_height)
    men_height_min = min(men_height)
    print(women_height_avg, women_height_max, women_height_min, men_height_avg, men_height_max, men_height_min)
    
if __name__=="__main__":
    output_info('data.csv')