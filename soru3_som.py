
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

som = MiniSom(x=8, y=8, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_batch(X, num_iteration=1000, verbose=True)

win_map = som.win_map(X)
mapped_labels = np.array([np.argmax(som.distance_map()[i]) for i in X])

plt.figure(figsize=(8, 8))
for i, (x, t) in enumerate(zip(X, mapped_labels)):
    plt.text(som.winner(x)[0]+.5, som.winner(x)[1]+.5, str(t),
             color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold', 'size': 11})
plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
plt.show()

