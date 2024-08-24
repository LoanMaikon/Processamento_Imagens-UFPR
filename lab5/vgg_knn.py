import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from glob import glob


def execute_simple():
    print("Executing simple extraction")

    train_path = "./train"
    test_path = "./test"

    train_dataset = image_dataset_from_directory(train_path, image_size=(224, 224), batch_size=32, shuffle=False)
    test_dataset = image_dataset_from_directory(test_path, image_size=(224, 224), batch_size=32, shuffle=False)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    raw_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = raw_model(x)
    output = Flatten()(x)

    model = Model(inputs, output)

    X_train = model.predict(train_dataset)
    X_test = model.predict(test_dataset)
    y_train = np.concatenate([y for _, y in train_dataset], axis=0)
    y_test = np.concatenate([y for _, y in test_dataset], axis=0)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print()

def execute_augmented_major():
    print("Executing augmented extraction using majority rule")

    train_path = "./train_augmented"
    test_path = "./test_augmented"

    raw_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    inputs = keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = raw_model(x)
    output = Flatten()(x)
    model = Model(inputs, output)

    test_images = glob(f"test/**/*.JPG", recursive=True)
    train_images = glob(f"{train_path}/**/*.jpg", recursive=True)

    X_train = []
    y_train = []
    for idx in range(len(train_images)):
        image_class = int(train_images[idx].split('/')[2])

        img = keras.preprocessing.image.load_img(train_images[idx], target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = preprocess_input(img_array)

        embeddings = model.predict(img_array, verbose=0)[0]

        X_train.append(embeddings)
        y_train.append(image_class)

    y_test = []
    images_groups = []
    for idx in range(len(test_images)):
        image_groups = []

        image_name = test_images[idx].split('/')[1:]
        y_test.append(int(image_name[0]))
        image_name = test_path + "/" + "/".join(image_name)

        for i in range(9):
            image_name_ = image_name[:-4] + f"_{i}.jpg"
            
            image_groups.append(image_name_)
        
        images_groups.append(image_groups)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    
    images_groups_embeddings = []
    for group in images_groups:
        group_embeddings = []
        for image in group:
            img = keras.preprocessing.image.load_img(image, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = preprocess_input(img_array)

            embeddings = model.predict(img_array, verbose=0)[0]

            group_embeddings.append(embeddings)
        
        images_groups_embeddings.append(group_embeddings)

    y_pred = []
    for image_group_idx in range(len(images_groups_embeddings)):
        X_test = images_groups_embeddings[image_group_idx]
        y_test_ = [y_test[image_group_idx]] * 9

        predictions = knn.predict(X_test)

        chosen_class = np.argmax(np.bincount(predictions))

        y_pred.append(chosen_class)
        
    print("Accuracy: ", accuracy_score(y_test, y_pred))


def main():
    execute_simple()
    execute_augmented_major()


if __name__ == "__main__":
    main()