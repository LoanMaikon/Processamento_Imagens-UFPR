import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from glob import glob
import cv2


def execute_simple():
    print("Executing simple extraction")
    train_path = "./train"
    test_path = "./test"

    train_images = glob(f"{train_path}/**/*.JPG", recursive=True)
    test_images = glob(f"{test_path}/**/*.JPG", recursive=True)

    y_train = [int(image.split('/')[2]) for image in train_images]
    y_test = [int(image.split('/')[2]) for image in test_images]

    X_train = []
    for image in train_images:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        lbp = local_binary_pattern(img, 8, 1, method="nri_uniform")
        hist, _ = np.histogram(lbp, bins=59, normed=True)

        X_train.append(hist)

    X_test = []
    for image in test_images:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        lbp = local_binary_pattern(img, 8, 1, method="nri_uniform")
        hist, _ = np.histogram(lbp, bins=59, normed=True)

        X_test.append(hist)
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))

def execute_augmented_major():
    print("Executing augmented extraction using majority rule")

    train_path = "./train_augmented"
    test_path = "./test_augmented"

    test_images = glob(f"test/**/*.JPG", recursive=True)
    train_images = glob(f"{train_path}/**/*.jpg", recursive=True)

    X_train = []
    y_train = []
    for idx in range(len(train_images)):
        image_class = int(train_images[idx].split('/')[2])

        img = cv2.imread(train_images[idx], cv2.IMREAD_GRAYSCALE)
        lbp = local_binary_pattern(img, 8, 1, method="nri_uniform")
        hist, _ = np.histogram(lbp, bins=59, normed=True)

        X_train.append(hist)
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
    
    images_groups_hists = []
    for group in images_groups:
        group_embeddings = []
        for image in group:
            image_class = int(train_images[idx].split('/')[2])

            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            lbp = local_binary_pattern(img, 8, 1, method="nri_uniform")
            hist, _ = np.histogram(lbp, bins=59, normed=True)

            group_embeddings.append(hist)
        
        images_groups_hists.append(group_embeddings)

    y_pred = []
    for image_group_idx in range(len(images_groups_hists)):
        X_test = images_groups_hists[image_group_idx]

        predictions = knn.predict(X_test)

        chosen_class = np.argmax(np.bincount(predictions))

        y_pred.append(chosen_class)
        
    print("Accuracy: ", accuracy_score(y_test, y_pred))


def main():
    execute_simple()
    execute_augmented_major()

if __name__ == "__main__":
    main()