from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
mnist.keys()

import numpy as np
X, y = mnist["data"], mnist["target"]

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

import numpy as np

def move_image_1px(X, dir):
    image = np.array(X).reshape(28, 28)
    shifted_image = np.zeros((28, 28))

    if dir == "right":
        part = image[:, :-1] 
        shifted_image[:, 1:] = part
        
    elif dir == "left":
        part = image[:, 1:]
        shifted_image[:, :-1] = part
        
    elif dir == "up":
        part = image[1:, :]
        shifted_image[:-1, :] = part

    elif dir == "down":
        part = image[:-1, :]
        shifted_image[1:, :] = part
        
    # Retorna a imagem deslocada (achatada para o formato original de 784)
    return shifted_image.flatten()

X_augmented = []
y_augmented = []
for i in range(0, len(X_train)):
    x_up = move_image_1px(X_train[i], "up")
    x_down = move_image_1px(X_train[i], "down")
    x_left = move_image_1px(X_train[i], "left")
    x_right = move_image_1px(X_train[i], "right")
    
    X_augmented.extend([x_up, x_down, x_left, x_right, X_train[i]])
    
    for i in range(0,5):
        y_augmented.append(y_train[i])
    
    
from sklearn.preprocessing import StandardScaler

X_aug_prep = StandardScaler().fit_transform(X_augmented)
X_prep = StandardScaler().fit_transform(X_train)


X_test_aug_prep = StandardScaler().fit_transform(X_augmented)
X_test_prep = StandardScaler().fit_transform(X_train)

from sklearn.linear_model import SGDClassifier

svc = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=1e-3, random_state=42)
svc.fit(X_aug_prep,y_augmented)
y_aug_pred = svc.predict(X_test_aug_prep)

svc.fit(X_prep,y)
y_pred = svc.predict(X_test_prep)

from sklearn.metrics import accuracy_score, f1_score

print("Modelo Normal -- Acuracia: ", accuracy_score(y, y_pred), " -- f1: ", f1_score(y,y_pred))
print("Modelo Aumentado -- Acuracia: ", accuracy_score(y, y_aug_pred), " -- f1: ", f1_score(y,y_aug_pred))