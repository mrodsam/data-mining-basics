import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from utils import plot_individual_history, evaluation, comparative, prediction
from dataset import load_data
from nn_models import simple_neural_network_model, neural_network_model, cnn_model, pretrained_model, cnn_model_avg, pretrained_model_conv, basic_cnn

tf.config.experimental.list_physical_devices('GPU')

augmentation = False
batch_size = 32

epochs = 60
only_save_model = False
models = []
histories = []

cv = False
num_folds = 10
acc_per_fold = []
loss_per_fold = []


def main():
    # Cargar datos
    global images, labels
    (train_images, train_labels), (test_images, test_labels), class_names = load_data()
    valid_images = test_images
    valid_labels = test_labels

    # Modelos
    model1 = simple_neural_network_model()
    models.append(model1)

    model2 = neural_network_model()
    models.append(model2)

    model3 = basic_cnn()
    models.append(model3)

    model4 = cnn_model()
    models.append(model4)

    model5 = cnn_model_avg()
    models.append(model5)

    model6 = pretrained_model()
    models.append(model6)

    model7 = pretrained_model_conv()
    models.append(model7)

    if only_save_model:
        exit(0)

    if cv:
        images = np.concatenate((train_images, test_images), axis=0)
        labels = np.concatenate((train_labels, test_labels), axis=0)

    kfold = KFold(n_splits=num_folds, shuffle=True)

    for index, model in enumerate(models):

        # Definir callbacks
        checkpoint = ModelCheckpoint("best_weights.h5",
                                     monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='max')

        stop = EarlyStopping(monitor='val_accuracy', patience=25, mode='max')

        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=15, min_lr=1e-7, verbose=1,
                                      mode="max")

        # Validación cruzada
        if cv:
            fold_no = 1
            for train, test in kfold.split(images, labels):
                print(f'Training for fold {fold_no}')
                history = model.fit(images[train], labels[train], epochs=epochs)

                scores = model.evaluate(images[test], labels[test], verbose=0)
                print(
                    f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} '
                    f'of {scores[1] * 100}%')
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])

                fold_no = fold_no + 1

            print('------------------------------------------------------------------------')
            print('Resultados por fold: ')
            for i in range(0, len(acc_per_fold)):
                print('------------------------------------------------------------------------')
                print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
            print('------------------------------------------------------------------------')
            print('Promedio:')
            print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
            print(f'> Loss: {np.mean(loss_per_fold)}')
            print('------------------------------------------------------------------------')
        else:
            # Generación de datos artificiales
            if augmentation:
                # Tipo de modificación
                train_datagen = keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )

                test_datagen = keras.preprocessing.image.ImageDataGenerator()

                train_generator = train_datagen.flow(
                    train_images,
                    train_labels,
                    batch_size=batch_size,
                )

                validation_generator = test_datagen.flow(
                    valid_images,
                    valid_labels,
                    batch_size=batch_size,
                )

                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    steps_per_epoch=len(train_images) // 32,
                    validation_data=validation_generator,
                    callbacks=[checkpoint, reduce_lr])
            else:
                history = model.fit(train_images, train_labels, epochs=epochs,
                                    validation_data=(valid_images, valid_labels), callbacks=[checkpoint, reduce_lr])

            histories.append(history)
            plot_individual_history(history, 'modelo'+ str(index + 1))

            print('Evaluación del último modelo: ')
            evaluation(model, 'modelo' + str(index + 1), test_images, test_labels)

            print('Evaluación del mejor modelo: ')
            model.load_weights('best_weights.h5')
            evaluation(model, 'modelo' + str(index + 1), test_images, test_labels)

            prediction(model, 'modelo' + str(index + 1), test_images, test_labels, class_names)
            os.remove('best_weights.h5')

    comparative(histories)


if __name__ == '__main__':
    main()


