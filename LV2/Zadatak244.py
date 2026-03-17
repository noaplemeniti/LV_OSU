import numpy as np
import matplotlib.pyplot as plt

def task():
    white = np.zeros((50, 50))
    black = np.ones((50, 50)) 
    horizontal1 = np.hstack((white, black))
    horizontal2 = np.hstack((black, white))
    img = np.vstack((horizontal1, horizontal2))
    plt.imshow(img, cmap="gray")
    plt.show()
    


if __name__=="__main__":
    task()
