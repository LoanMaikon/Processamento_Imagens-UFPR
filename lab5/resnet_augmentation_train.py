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
from glob import glob
import shutil
import sys
import os

def train(train_path, validation_path, model_name):
    train_dataset = image_dataset_from_directory(train_path, image_size=(224, 224), batch_size=32, shuffle=True, seed=42)
    validation_dataset = image_dataset_from_directory(validation_path, image_size=(224, 224), batch_size=32, shuffle=False)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    num_classes = 9

    raw_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    raw_model.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = raw_model(x)
    output = GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.1)(output)
    predictions = Dense(num_classes)(x)
    model = Model(inputs, predictions)

    # Treinamento da camada densa
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    callbacks = [EarlyStopping(patience=10),
                 ModelCheckpoint(filepath=model_name, save_weights_only=False, save_best_only=True, verbose=1, monitor="val_accuracy", mode="max")]
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=50, callbacks=callbacks, verbose=1)

    # Treinamento de toda a rede
    # raw_model.trainable = True
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # callbacks = [EarlyStopping(patience=15),
    #                 ModelCheckpoint(filepath=model_name, save_weights_only=False, save_best_only=True, verbose=1, monitor="val_accuracy", mode="max")]
    # history = model.fit(train_dataset, validation_data=validation_dataset, epochs=2, callbacks=callbacks, verbose=1, batch_size=2)


def test(test_path, model_name):
    best_model = tf.keras.models.load_model(model_name)

    test_images = glob(f"test/**/*.JPG", recursive=True)
    
    real_classes = []
    images_groups = []
    for idx in range(len(test_images)):
        image_groups = []

        image_name = test_images[idx].split('/')[1:]
        real_classes.append(int(image_name[0]))
        image_name = test_path + "/" + "/".join(image_name)

        for i in range(9):
            image_name_ = image_name[:-4] + f"_{i}.jpg"
            
            image_groups.append(image_name_)
        
        images_groups.append(image_groups)
    
    images_confidences = []
    for group in images_groups:
        confidences = []
        for image in group:
            img = keras.preprocessing.image.load_img(image, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = best_model.predict(img_array)
            scores = tf.nn.softmax(predictions[0])

            confidences.append(scores)
        
        images_confidences.append(confidences)

    predicted_classes = []
    for image_idx in range(len(images_confidences)):
        sum_confidences = [0] * 9

        for image_splitted_idx in range(9):
            for class_idx in range(9):
                sum_confidences[class_idx] += images_confidences[image_idx][image_splitted_idx][class_idx]
        
        sum_confidences = [sum_confidence.numpy() for sum_confidence in sum_confidences]

        predicted_classes.append(sum_confidences.index(max(sum_confidences)))

    score = accuracy_score(real_classes, predicted_classes)

    print(f"Accuracy: {score}")


if __name__ == "__main__":
    train_path = "./train_augmented"
    test_path = "./test_augmented"
    validation_path = "./validation_augmented"

    if len(sys.argv) < 3:
        print("Use: python resnet_augmentation_train.py <train|test> <nome_do_modelo>")
        exit(1)

    model_name = sys.argv[2]

    if sys.argv[1] == "train":
        if os.path.exists(model_name):
            print(f"O modelo {model_name} já existe.")
            exit(1)
        
        train(train_path, validation_path, model_name)
    elif sys.argv[1] == "test":
        test(test_path, model_name)
