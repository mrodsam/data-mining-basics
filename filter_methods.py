from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np
import sklearn_relief as relief
from ReliefF import ReliefF


def chi_squared(X, Y):
    print('*********************** Chi2 ********************************')
    fs = SelectKBest(score_func=chi2, k='all')
    fit = fs.fit(X, Y)

    np.set_printoptions(precision=3)

    for i in range(len(fs.scores_)):
        print('Variable %d: %f' % (i, fs.scores_[i]))
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.savefig('chi2.png')
    plt.show()


def mutual_info(X, Y):
    print('*********************** Información mutua ********************************')
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fit = fs.fit(X, Y)

    np.set_printoptions(precision=3)

    for i in range(len(fs.scores_)):
        print('Variable %d: %f' % (i, fs.scores_[i]))
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.savefig('mutualinfo.png')
    plt.show()


def relief_alg(X, Y):
    print('*********************** ReliefF ********************************')
    for i in range(10, 60, 10):
        print('k = ', str(i))
        fs = ReliefF(n_neighbors=i, n_features_to_keep=5)
        fs.fit(X, Y)
        print(fs.feature_scores)

    fs = ReliefF(n_neighbors=10, n_features_to_keep=5)
    fs.fit(X, Y)
    for i in range(len(fs.feature_scores)):
        print('Variable %d: %f' % (i, fs.feature_scores[i]))
    plt.bar([i for i in range(len(fs.feature_scores))], fs.feature_scores)
    plt.savefig('relieff.png')
    plt.show()

