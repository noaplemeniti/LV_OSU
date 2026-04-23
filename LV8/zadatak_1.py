import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import cv2

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.imshow(x_train[0])
plt.show()


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model_path = "cnn_lv8.keras"
print(os.path.exists(model_path))
if os.path.exists(model_path):
model = keras.models.load_model("cnn_lv8.keras")
else:
# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss ="categorical_crossentropy",
optimizer ="adam",
metrics =["accuracy",])
batch_size = 16
epochs = 20

# TODO: provedi ucenje mreze
history = model.fit(x_train_s, y_train_s, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
model.save("cnn_lv8.keras")


# TODO: Prikazi test accuracy i matricu zabune
predictions = model.predict(x_test_s)
score = model.evaluate(x_test_s, y_test_s)
prediction_values = np.argmax(predictions, axis=-1)
print(prediction_values)

# TODO: spremi model
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
print(y_test)
print(prediction_values)
cm = confusion_matrix(y_test, prediction_values)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

incorrect_indexes = []

for idx, (y_true, y_pred) in enumerate(zip(y_test, prediction_values)):
if y_true != y_pred:
incorrect_indexes.append(idx)

for i in range(1, 5):
plt.subplot(2, 2, i)
plt.imshow(x_test[incorrect_indexes[i-1]])
plt.title(prediction_values[incorrect_indexes[i-1]])


plt.show()


test_image = cv2.imread("test.png")
test_image = cv2.resize(test_image, (28, 28))
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)


test_image = test_image.astype("float32") / 255
test_image = np.expand_dims(test_image, axis=0)
test_image = np.expand_dims(test_image, axis=-1)

test_img_pred = model.predict(test_image)

test_img_pred_val = np.argmax(test_img_pred, axis=-1)

print(test_img_pred_val)