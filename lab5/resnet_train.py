import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from pathlib import Path
import shutil
import sys
import os

def train(train_path, validation_path, model_name):
    train_dataset = image_dataset_from_directory(train_path, image_size=(224, 224), batch_size=32, shuffle=True, seed=42)
    validation_dataset = image_dataset_from_directory(validation_path, image_size=(224, 224), batch_size=32, shuffle=False)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    num_classes = 9

    data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                             tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                                             tf.keras.layers.experimental.preprocessing.RandomContrast(0.2, 0.2)])

    raw_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    raw_model.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = raw_model(x)
    output = GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0)(output)
    predictions = Dense(num_classes)(x)
    model = Model(inputs, predictions)

    # Treinamento da camada densa
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    callbacks = [EarlyStopping(patience=10),
                 ModelCheckpoint(filepath=model_name, save_weights_only=False, save_best_only=True, verbose=1, monitor="val_accuracy", mode="max")]
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=100, callbacks=callbacks, verbose=1)

    # Treinamento de toda a rede
    # raw_model.trainable = True
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # callbacks = [EarlyStopping(patience=15),
    #                 ModelCheckpoint(filepath=model_name, save_weights_only=False, save_best_only=True, verbose=1, monitor="val_accuracy", mode="max")]
    # history = model.fit(train_dataset, validation_data=validation_dataset, epochs=50, callbacks=callbacks, verbose=1)


def test(test_path, model_name):
    test_dataset = image_dataset_from_directory(test_path, image_size=(224, 224), batch_size=32, shuffle=False)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    best_model = tf.keras.models.load_model(model_name)
    predict = best_model.predict(test_dataset)

    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_predict = np.argmax(predict, axis=1)

    accuracy = accuracy_score(y_true, y_predict)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    train_path = "./train"
    test_path = "./test"
    validation_path = "./validation"

    model_name = sys.argv[2]

    if sys.argv[1] == "train":
        if os.path.exists("model"):
            print("A pasta model já existe")
            exit(1)
        train(train_path, validation_path, model_name)
    elif sys.argv[1] == "test":
        test(test_path, model_name)
