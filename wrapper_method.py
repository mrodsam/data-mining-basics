from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def rfe(X, Y):
    print('*********************** RFE ********************************')

    models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]

    for model in models:
        print(f'Modelo: {model}')
        fs = RFE(model, n_features_to_select=2)
        fit = fs.fit(X, Y)
        print("Num Features: %s" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)
