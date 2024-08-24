from utils import _get_classes, _get_folder_name, _get_images, _make_histograms, _load_histograms, \
                  _choose_class, _get_ac, _get_validation_data
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
import cv2
import os


def _input_treatment(args):
    if len(args) < 2 or len(args) > 3:
        print("Usage: python3 Kvalidation.py <dataset_folder> <validation_size>")
        exit(1)
    
    images_folder = args[1]
    if len(args) == 3:
        validation_size = float(args[2])
    else:
        validation_size = 0.4

    if os.path.isdir(images_folder) is False:
        print(f"{images_folder} is not a valid folder")
        exit(1)

    return images_folder, validation_size


'''
CLASS ID: CLASS
1: Bart
2: Homer
3: Lisa
4: Marge
5: Maggie
'''
if __name__ == "__main__":
    images_folder, validation_size = _input_treatment(sys.argv)
    image_paths = _get_images(images_folder)

    classes = _get_classes(image_paths)

    folder_name = _get_folder_name(images_folder)

    os.makedirs("jsons", exist_ok=True)
    JSON_PATH = f"jsons/histograms_{folder_name}.json"
    if not os.path.exists(JSON_PATH):
        histograms = _make_histograms(image_paths, JSON_PATH)
    histograms = _load_histograms(JSON_PATH)

    np.random.seed(69)
    validation_histograms, validation_classes = _get_validation_data(histograms, classes, validation_size)

    K = [1, 2, 3, 4]
    methods = [1, 2, 3, 4, 5]


    for k in K:
        acs = []
        for method in methods:

            y_test, y_pred = [], []
            # Comparing each image with the others
            for i in range(0, len(validation_histograms), 3):
                comparisons = []

                for j in range(0, len(validation_histograms), 3):
                    if i == j:
                        continue

                    ac = _get_ac(validation_histograms[i:i+3], validation_histograms[j:j+3], method)
                    
                    comparisons.append(ac)
                
                if method in [2, 4]: # Sorting in descending order
                    sorted_comparisons, sorted_classes = zip(*sorted(zip(comparisons, validation_classes.copy()), reverse=True))
                elif method in [1, 3, 5]: # Sorting in ascending order
                    sorted_comparisons, sorted_classes = zip(*sorted(zip(comparisons, validation_classes.copy()), reverse=False))

                y_test.append(validation_classes[i//3])
                y_pred.append(_choose_class(sorted_classes[:k]))

            cm = confusion_matrix(y_test, y_pred)
            ac = np.trace(cm) / np.sum(cm)

            acs.append(ac)
        
        ac = np.mean(acs)

        print(f"For K = {k}, the Mean Accuracy is {ac}")