import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

colors = ["red", "green", "blue", "yellow", "orange"]
cmp = ListedColormap(colors)

#plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmp)
#plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cmp, marker='x')
#plt.show()

linearModel = lm.LogisticRegression()
sc = MinMaxScaler()

x_train_n = sc.fit_transform(x_train)
x_test_n = sc.transform(x_test)

linearModel.fit(x_train_n, y_train)

y_test_p = linearModel.predict(x_test_n)


plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cmp)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test_p, cmap=cmp, marker='.')
#plt.show()

coef = linearModel.coef_[0]
intercept = linearModel.intercept_[0]

x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(intercept + coef[0]*x_vals) / coef[1]

plt.plot(x_vals, y_vals, color='black')
plt.show()

print ("Tocnost: " , accuracy_score(y_test, y_test_p))

cm = confusion_matrix(y_test, y_test_p)
print("Matrica zabune : ", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()
# report
print (classification_report(y_test, y_test_p))

y_test_true_false = (y_test == y_test_p).astype(int)

plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test_true_false, cmap=cmp)
plt.show()
