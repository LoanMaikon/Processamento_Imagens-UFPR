from sklearn.model_selection import train_test_split
from glob import glob
import os
import cv2


def augment(images, classes, path):
    os.makedirs(path, exist_ok=True)

    for idx, image in enumerate(images):
        os.makedirs(f'{path}/{classes[idx]}', exist_ok=True)

        img = cv2.imread(image)
        total_height = img.shape[0]
        total_width = img.shape[1]

        # Determina as dimensões das sub-imagens
        segment_width = total_width // 3
        segment_height = total_height // 3
        h = segment_height
        w = segment_width

        count = 0
        for i in range(3):
            for j in range(3):
                x = j*segment_width
                y = i*segment_height
                roi = img[y:y+h, x:x+w]
                cv2.imwrite(f"{path}/{classes[idx]}/{image.split('/')[1][:-4]}_{count}.jpg", roi)
                count += 1

def main():
    if os.path.exists('train') or os.path.exists('test'):
        print('Os diretórios "train" ou "test" já existem.')
        exit(1)

    images = glob('macroscopic0/**/*.JPG', recursive=True)

    # Determinando as classes
    classes = []
    for image in images:
        imageClass = int(image.split('/')[1][0:2])
        
        classes.append(imageClass - 1)
    
    classes_ids = list(set(classes))

    # Dividindo os dados em treino e teste
    train_images, test_images = [], []
    train_classes, test_classes = [], []
    validation_images, validation_classes = [], []
    test_plus_validation_images, test_plus_validation_classes = [], []

    for class_id in classes_ids:
        id_images = [images[idx] for idx in range(len(images)) if classes[idx] == class_id]
        id_classes = [class_id] * len(id_images)

        train_images_class, temp_images_class, train_classes_class, temp_classes_class = train_test_split(id_images, id_classes, test_size=0.5, random_state=42)

        test_images_class, validation_images_class, test_classes_class, validation_classes_class = train_test_split(temp_images_class, temp_classes_class, test_size=0.4, random_state=42)

        train_images.extend(train_images_class)
        test_images.extend(test_images_class)
        train_classes.extend(train_classes_class)
        test_classes.extend(test_classes_class)
        validation_images.extend(validation_images_class)
        validation_classes.extend(validation_classes_class)
        test_plus_validation_images.extend(temp_images_class)
        test_plus_validation_classes.extend(temp_classes_class)

    # Salvando as imagens de treino
    for train_idx in range(len(train_images)):
        os.makedirs(f'train/{train_classes[train_idx]}', exist_ok=True)

        os.system(f'cp {train_images[train_idx]} train/{train_classes[train_idx]}/')

    # Salvando as imagens de teste
    for test_idx in range(len(test_images)):
        os.makedirs(f'test/{test_classes[test_idx]}', exist_ok=True)

        os.system(f'cp {test_images[test_idx]} test/{test_classes[test_idx]}/')
    
    # Salvando as imagens de validação
    for validation_idx in range(len(validation_images)):
        os.makedirs(f'validation/{validation_classes[validation_idx]}', exist_ok=True)

        os.system(f'cp {validation_images[validation_idx]} validation/{validation_classes[validation_idx]}/')

    # Salvando as imagens de teste + validação
    for test_plus_validation_idx in range(len(test_plus_validation_images)):
        os.makedirs(f'test_plus_validation/{test_plus_validation_classes[test_plus_validation_idx]}', exist_ok=True)

        os.system(f'cp {test_plus_validation_images[test_plus_validation_idx]} test_plus_validation/{test_plus_validation_classes[test_plus_validation_idx]}/')

    # Criando as imagens segmentadas em 3x3
    augment(train_images, train_classes, 'train_augmented')
    augment(test_images, test_classes, 'test_augmented')
    augment(validation_images, validation_classes, 'validation_augmented')
    augment(test_plus_validation_images, test_plus_validation_classes, 'test_plus_validation_augmented')

if __name__ == '__main__':
    main()