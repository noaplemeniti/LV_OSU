import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

import sklearn.linear_model as lm
from sklearn.preprocessing import MinMaxScaler

from sklearn . metrics import accuracy_score, classification_report
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

y_flat = y.flatten()

class_counts = [np.count_nonzero(y_flat == 0), np.count_nonzero(y_flat == 1), np.count_nonzero(y_flat == 2)]

plt.bar(['Adelie', 'Chinstrap', 'Gentoo'], class_counts)
plt.show()

sc = MinMaxScaler()
x_train_n = sc.fit_transform(X_train)

linearModel = lm.LogisticRegression(max_iter=1000)
linearModel.fit(x_train_n, y_train.ravel())

coef = linearModel.coef_[0]
intercept = linearModel.intercept_[0]

plot_decision_regions(x_train_n, y_train.ravel(), linearModel)
plt.xlabel('Bill length (scaled)')
plt.ylabel('Flipper length (scaled)')
plt.legend()
plt.show()


x_test_n = sc.transform(X_test)

y_pred = linearModel.predict(x_test_n)

print (" Tocnost : " , accuracy_score ( y_test , y_pred ) )
# matrica zabune
cm = confusion_matrix ( y_test , y_pred )
print (" Matrica zabune : " , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_pred ) )
disp . plot ()
plt . show ()
# report
print ( classification_report ( y_test , y_test ) )