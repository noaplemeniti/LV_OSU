from matplotlib import image
from matplotlib import pyplot
import numpy as np

def task():
    img = image.imread("road.jpg")
    print(img)
    print(img.shape)
    augmented_img = img.copy()

    augmented_img[::3, ::3] = [255, 255, 255]

    pyplot.imshow(augmented_img)
    pyplot.show()

    bright_img = np.clip(img * 3, 0, 255).astype(np.uint8)
    pyplot.imshow(bright_img)
    pyplot.show()

    M = img.shape[0] // 2
    N = img.shape[1] // 2

    tiles = [img[x:x+M, y:y+N] 
             for x in range(0, img.shape[0], M) 
             for y in range(0, img.shape[1], N)]
    tiles[0] = tiles[0] * 0
    tiles[2] = tiles[2] * 0
    tiles[3] = tiles[3] * 0
    top = np.hstack((tiles[0], tiles[1]))
    bottom = np.hstack((tiles[2], tiles[3]))
    new_img = np.vstack((top, bottom))

    pyplot.imshow(new_img)
    pyplot.show()

    rotated = np.rot90(img, 3)
    pyplot.imshow(rotated)
    pyplot.show()

    mirrored = np.fliplr(img)
    pyplot.imshow(mirrored)
    pyplot.show()

    


if __name__ == "__main__":
    task()