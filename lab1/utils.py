from sklearn.metrics import confusion_matrix
from glob import glob
import numpy as np
import json
import cv2


def _get_images(folder, extension=".bmp"):
    return glob(f"{folder}/*{extension}")

def _get_folder_name(folder):
    folder_name = folder.split("/")
    for name in folder_name:
        if name == "" or name == ".":
            folder_name.remove(name)
    folder_name = "_".join(folder_name)

    return folder_name

def _get_classes(image_paths):
    classes = []

    for image_path in image_paths:
        image_name = image_path.split("/")[-1]
        
        if image_name.startswith("b"):
            classes.append(1)
        elif image_name.startswith("h"):
            classes.append(2)
        elif image_name.startswith("l"):
            classes.append(3)
        elif image_name.startswith("mg"):
            classes.append(5)
        elif image_name.startswith("m"):
            classes.append(4)

    return classes

def _make_histograms(image_paths, json_path):
    histograms = []

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        histogram1 = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram1[0] = 0
        histogram1[255] = 0
        cv2.normalize(histogram1, histogram1)

        histogram2 = cv2.calcHist([image], [1], None, [256], [0, 256])
        histogram2[0] = 0
        histogram2[255] = 0
        cv2.normalize(histogram2, histogram2)

        histogram3 = cv2.calcHist([image], [2], None, [256], [0, 256])
        histogram3[0] = 0
        histogram3[255] = 0
        cv2.normalize(histogram3, histogram3)

        histogram1 = [[float(value)] for value in histogram1]
        histogram2 = [[float(value)] for value in histogram2]
        histogram3 = [[float(value)] for value in histogram3]

        histograms.append(histogram1)
        histograms.append(histogram2)
        histograms.append(histogram3)

    with open(json_path, "w") as file:
        json.dump(histograms, file)

def _load_histograms(json_path):
    histograms = json.load(open(json_path))

    histograms = [np.array(histogram, dtype=np.float32) for histogram in histograms]

    return histograms

def _compareHistEuclidean(histogram1, histogram2):
    return np.linalg.norm(histogram1 - histogram2)

def _choose_class(classes):
    unique_classes = list(set(classes))

    max_class = unique_classes[0]
    max_count = classes.count(max_class)
    for unique_class in unique_classes[1:]:
        count = classes.count(unique_class)
        if count > max_count or (count == max_count and classes.index(unique_class) < classes.index(max_class)):
            max_class = unique_class
            max_count = count

    return max_class

def _get_validation_data(histograms, classes, validation_size):
    validation_histograms = []
    validation_classes = []

    for _ in range(int(len(histograms) // 3 * validation_size)):
        index = np.random.randint(0, len(histograms) // 3)

        validation_histograms.append(histograms[3*index])
        validation_histograms.append(histograms[3*index + 1])
        validation_histograms.append(histograms[3*index + 2])
        validation_classes.append(classes[index])

    return validation_histograms, validation_classes

def _get_ac(histograms1, histograms2, method):
    acs = []

    for i in range(3):
        match method:
            case 1: # Euclidean distance
                acs.append(_compareHistEuclidean(histograms1[i], histograms2[i]))
            case 2: # Correlation
                acs.append(cv2.compareHist(histograms1[i], histograms2[i], cv2.HISTCMP_CORREL))
            case 3: # Chi-Square
                acs.append(cv2.compareHist(histograms1[i], histograms2[i], cv2.HISTCMP_CHISQR))
            case 4: # Intersection
                acs.append(cv2.compareHist(histograms1[i], histograms2[i], cv2.HISTCMP_INTERSECT))
            case 5: # Bhattacharyya distance
                acs.append(cv2.compareHist(histograms1[i], histograms2[i], cv2.HISTCMP_BHATTACHARYYA))
    
    if method in [2, 4]:
        return np.max(acs)
    elif method in [1, 3, 5]:
        return np.min(acs)
    return None
