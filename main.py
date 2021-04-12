import numpy as np
import pandas as pd
import random
from filter_methods import chi_squared, mutual_info, relief_alg
from wrapper_method import rfe
from pca import pca


def random_bool():
    return bool(random.getrandbits(1))


def generate_data():
    size = 100
    a = np.random.choice([0, 1], size=size)
    b = np.random.choice([0, 1], size=size)
    c = np.random.choice([0, 1], size=size)

    d = b ^ c
    e = np.random.choice([0, 1], size=size)

    y = a ^ b ^ c

    data = {'X1': a, 'X2': b, 'X3': c, 'X4': d, 'X5': e, 'Y': y}

    df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'Y'])

    return df


if __name__ == '__main__':
    dataframe = generate_data()
    array = dataframe.values
    X = array[:, :-1]
    Y = array[:, -1]

    # Técnicas de filtrado
    chi_squared(X, Y)
    mutual_info(X, Y)
    relief_alg(X, Y)

    # Técnica de envoltura
    rfe(X, Y)

    #Análisis de componentes principales
    pca(X, Y)



