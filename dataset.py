import re
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras

dirname = os.path.join(os.getcwd(), 'anuka1200')
imgpath = dirname + os.sep


def load_data():
    images = []
    dircount = []
    prev_root = ''
    cant = 0

    print('leyendo imágenes de ', imgpath)

    for root, dirnames, filenames in os.walk(imgpath):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant = cant + 1
                filepath = os.path.join(root, filename)
                image = cv2.imread(filepath)
                images.append(image)
                if prev_root != root:
                    print(root, cant)
                    prev_root = root
                    dircount.append(cant)
                    cant = 0

    dircount.append(cant)

    dircount = dircount[1:]
    dircount[0] = dircount[0] + 1
    print("Imagenes en cada directorio", dircount)
    print('suma Total de imagenes en subdirs:', sum(dircount))

    labels = []
    index = 0
    for cantidad in dircount:
        for i in range(cantidad):
            labels.append(index)
        index = index + 1

    y = np.array(labels)
    X = np.array(images)

    class_names = ['Kunzea', 'Lepto']

    # Mezclar y crear los grupos de entrenamiento y testing
    train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

    # # Mezclar y crear los grupos de entrenamiento y validación
    # train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels,
    #                                                                           test_size=0.25)
    print('Datos de entrenamiento : ', train_images.shape, train_labels.shape)
    # print('Datos de validación : ', test_images.shape, test_labels.shape)
    print('Datos de test : ', test_images.shape, test_labels.shape)

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    # valid_images = valid_images.astype('float32')
    train_images = train_images / 255.
    test_images = test_images / 255.
    # valid_images = valid_images / 255.
    # return (train_images, train_labels), (test_images, test_labels), (valid_images, valid_labels), class_names
    return (train_images, train_labels), (test_images, test_labels), class_names
