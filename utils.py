import numpy as np
import matplotlib.pyplot as plt

show = False


def comparative(histories):

    plt.figure(figsize=(8, 8))
    for i, history in enumerate(histories):
        if i == 4 or i == 5:  # Modelos a saltar
            pass
        else:
            plt.plot(history.history['val_accuracy'], label='Modelo '+str(i + 1))
            plt.legend(loc='lower right')
            plt.xlabel('Número de épocas')
            plt.ylabel('Precisión')
            plt.savefig('comparativa-acc.png')

    plt.figure(figsize=(8, 8))
    for i, history in enumerate(histories):
        if i == 4 or i == 5:  # Modelos a saltar
            pass
        else:
            plt.plot(history.history['val_loss'], label='Modelo ' + str(i + 1))
            plt.legend(loc='lower right')
            plt.xlabel('Número de épocas')
            plt.ylabel('Pérdidas')
            plt.savefig('comparativa-loss.png')

    if show:
        plt.show()


def plot_individual_history(history, modeldir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.plot(acc, label='Entrenamiento')
    plt.plot(val_acc, label='Validación')
    plt.legend(loc='lower right')
    plt.xlabel('Número de épocas')
    plt.ylabel('Precisión')
    plt.savefig(modeldir + '/' + modeldir + '-acc.png')

    plt.figure(figsize=(8, 8))
    plt.plot(loss, label='Entrenamiento')
    plt.plot(val_loss, label='Validación')
    plt.legend(loc='upper right')
    plt.xlabel('Número de épocas')
    plt.ylabel('Entropía cruzada categórica dispersa')
    plt.savefig(modeldir+'/'+modeldir+'-loss.png')

    if show:
        plt.show()


def evaluation(model, modeldir, test_images, test_labels):

    # Evaluación
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy: ', test_acc)
    with open('prediction.txt', 'a') as pred_file:
        pred_file.write(modeldir+': '+str(test_acc) + '\n')


def prediction(model, modeldir, test_images, test_labels, class_names):
    predictions = model.predict(test_images)
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.savefig(modeldir+'/muestras.png')
    if show:
        plt.show()


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap='gray')

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')