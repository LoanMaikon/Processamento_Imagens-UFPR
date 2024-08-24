from utils import _get_classes, _get_folder_name, _get_images, _make_histograms, _load_histograms, \
                  _choose_class, _get_ac
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
import cv2
import os

def _input_treatment(args):
    if len(args) < 3 or len(args) > 4:
        print("Usage: python3 histograma.py <method> <dataset_folder> <K>")
        exit(1)
    
    method = int(args[1])
    images_folder = args[2]
    if len(args) == 4:
        K = int(args[3])
    else:
        K = 1

    if method < 1 or method > 5:
        print("Method must be between 1 and 5")
        exit(1)

    if os.path.isdir(images_folder) is False:
        print(f"{images_folder} is not a valid folder")
        exit(1)

    return method, images_folder, K


'''
CLASS ID: CLASS
1: Bart
2: Homer
3: Lisa
4: Marge
5: Maggie
'''
if __name__ == "__main__":
    method, images_folder, K = _input_treatment(sys.argv)

    image_paths = _get_images(images_folder)
    classes = _get_classes(image_paths)

    folder_name = _get_folder_name(images_folder)

    os.makedirs("jsons", exist_ok=True)
    JSON_PATH = f"jsons/histograms_{folder_name}.json"
    if not os.path.exists(JSON_PATH):
        histograms = _make_histograms(image_paths, JSON_PATH)
    histograms = _load_histograms(JSON_PATH)

    y_test, y_pred = [], []
    # Comparing each image with the others
    for i in range(0, len(histograms), 3):
        comparisons = []

        for j in range(0, len(histograms), 3):
            if i == j:
                continue

            ac = _get_ac(histograms[i:i+3], histograms[j:j+3], method)
            
            comparisons.append(ac)
        
        if method in [2, 4]: # Sorting in descending order
            sorted_comparisons, sorted_classes = zip(*sorted(zip(comparisons, classes.copy()), reverse=True))
        elif method in [1, 3, 5]: # Sorting in ascending order
            sorted_comparisons, sorted_classes = zip(*sorted(zip(comparisons, classes.copy()), reverse=False))

        y_test.append(classes[i//3])
        y_pred.append(_choose_class(sorted_classes[:K]))

    cm = confusion_matrix(y_test, y_pred)
    ac = np.trace(cm) / np.sum(cm)

    print(f"Acurácia: {ac}\n")
    print(f"Matriz de confusão: \n{cm}")
