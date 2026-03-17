import matplotlib.pyplot as plt
import numpy as np

def draw_func():
    x = np.array([1, 3, 2, 1])
    y = np.array([1, 1, 5, 1])
    plt . plot (x, y, 'b', linewidth =1, marker =".", markersize =5)
    plt.xlim(0,4)
    plt.ylim(0,10)
    line1 = np.sqrt((2-1)**2+(5-1)**2)
    line2 = np.sqrt((2-3)**2+(5-1)**2)
    print(line1, line2)
    plt.show()

if __name__=="__main__":
    draw_func()