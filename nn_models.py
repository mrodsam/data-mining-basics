from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16

input_shape = (100, 100, 3)
learning_rate = 0.001


def simple_neural_network_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.save('model1.h5')

    return model


def neural_network_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    model.save('model2.h5')

    return model


def basic_cnn():
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), activation='relu', input_shape=input_shape,
                            padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.save('model3.h5')

    return model


def cnn_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), activation='relu', input_shape=input_shape,
                            padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.save('model4.h5')

    return model


def cnn_model_avg():
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), activation='relu', input_shape=input_shape,
                            padding='same'),
        keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.save('model5.h5')

    return model


def pretrained_model():
    vgg16model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')

    for layer in vgg16model.layers:
        layer.trainable = False

    vgg16model.summary()

    model = keras.models.Sequential([
        vgg16model,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.save('model6.h5')

    return model


def pretrained_model_conv():
    vgg16model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')

    for layer in vgg16model.layers[:15]:
        layer.trainable = False

    vgg16model.summary()

    model = keras.models.Sequential([
        vgg16model,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.save('model7.h5')

    return model
