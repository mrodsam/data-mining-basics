import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def pca(X, Y):
    print('*********************** PCA ********************************')

    pca = PCA()
    pca.fit(X)

    print('Varianza explicada: %s' % pca.explained_variance_ratio_)
    print('Componentes: %s' % pca.n_components_)

    xi = np.arange(1, 6, step=1)
    plt.plot(xi, np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xticks(np.arange(1, 6, step=1))
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza explicada acumulada')
    plt.grid(axis='x')
    plt.savefig('pca.png')
    plt.show()
