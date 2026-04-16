import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_2.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters = 4, n_init = 10)

km.fit(img_array_aprox)

labels = km.labels_

centers = km.cluster_centers_

img_array_aprox = centers[labels]

img_array_aprox = np.reshape(img_array_aprox, (w, h, d))
img_uint8 = (img_array_aprox * 255).astype(np.uint8)

k_values = range(1, 6)
inertias = []

plt.figure()
plt.title("Nova slika")
plt.imshow(img_uint8)
plt.tight_layout()
plt.show()

binary_img = (labels==0).astype(np.uint8)
binary_img = np.reshape(binary_img, (w,h))

print(binary_img)

plt.imshow(binary_img, cmap = 'Greys')
plt.show()

img_array_aprox = img_array.copy()

for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(img_array_aprox)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(k_values, inertias, marker='o')
plt.show()