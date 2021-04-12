# -*- coding: utf-8 -*-
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import model_selection
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import paired_ttest_5x2cv


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lda = LinearDiscriminantAnalysis()
    knn = KNeighborsClassifier()

    print('Métodos basados en validación cruzada: ')
    ten_fold_cross_validation(lda, knn, X, y)
    method_paired_ttest_5x2cv(lda, knn, X, y)

    knn.fit(X_train, y_train)
    lda.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    y_pred_lda = lda.predict(X_test)

    method_test_mcnemar(y_test, y_pred_knn, y_pred_lda)

    print('Evaluación individual de cada clasificador sobre el conjunto de pruebas: ')
    print('K vecinos más cercanos')
    knn_acc = knn.score(X_test, y_test)
    print(classification_report(y_test, y_pred_knn))
    print('Precisión: %.2f%%' % (knn_acc * 100))

    print('Análisis discriminante lineal')
    lda_acc = lda.score(X_test, y_test)
    print(classification_report(y_test, y_pred_lda))
    print('Precisión: %.2f%%' % (lda_acc * 100))


def ten_fold_cross_validation(lda, knn, X, y):
    k_fold_val = model_selection.KFold(n_splits=10, shuffle=True)
    result_knn = model_selection.cross_val_score(knn, X, y, cv=k_fold_val, scoring='accuracy')
    result_lda = model_selection.cross_val_score(lda, X, y, cv=k_fold_val, scoring='accuracy')

    print('Validación cruzada de 10 iteraciones')
    print("KNN: %f (%f)" % (result_knn.mean(), result_knn.std()))
    print("LDA: %f (%f)" % (result_lda.mean(), result_lda.std()))
    print()


def method_paired_ttest_5x2cv(lda, knn, X, y):
    t, p = paired_ttest_5x2cv(knn, lda, X, y, random_seed=1)

    print('Test t de Student pareado sobre 5x2CV')
    print('Estadístico t: ', t)
    print('Valor p: ', p)

    # Interpretación del resultado con un nivel de significatividad α = 0.05
    alpha = 0.05
    if p > alpha:
        print('Se acepta la hipótesis nula: ambos modelos tienen el mismo rendimiento.')
    else:
        print('Se rechaza la hipótesis nula: los modelos no tienen el mismo rendimiento.')
    print()

def method_test_mcnemar(y_test, y_pred_knn, y_pred_lda):
    table = mcnemar_table(y_test, y_pred_knn, y_pred_lda)
    print('Test de McNemar')
    print(table)

    chi2, p = mcnemar(table, corrected=True)
    print('Estadístico chi-cuadrado:', chi2)
    print('Valor p:', p)

    # Interpretación del resultado con un nivel de significatividad α = 0.05
    alpha = 0.05
    if p > alpha:
        print('Se acepta la hipótesis nula: ambos modelos tienen el mismo rendimiento.')
    else:
        print('Se rechaza la hipótesis nula: los modelos no tienen el mismo rendimiento.')
    print()


if __name__ == '__main__':
    main()


