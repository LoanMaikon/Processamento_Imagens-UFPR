from glob import glob
import numpy as np
import cv2
import sys
import os


'''
Add Noise with a given probability
'''
def sp_noise(image, prob):
        
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


'''
Generate noisy images with salt and pepper noise and save them to the images directory.
'''
def generate_noise_images(original_image, probabilities=[0.01, 0.02, 0.05, 0.07, 0.1], num_images=10):
    os.makedirs("images", exist_ok=True)

    for prob in probabilities:
        os.makedirs(f"images/noise{prob}", exist_ok=True)
        for i in range(num_images):
            noisy_image = sp_noise(original_image, prob)
            cv2.imwrite(f"images/noise{prob}/noise{i}.png", noisy_image)


'''
Print initial PSNR for all images
'''
def print_psnr(images, images_paths, original_image, num_images, probabilities):
    print(f"MAX PSNR = {cv2.PSNR(original_image, original_image)}")

    mean_psnr = [0] * len(probabilities)

    for i, image in enumerate(images):
        if str(probabilities[0]) in images_paths[i]:
            mean_psnr[0] += cv2.PSNR(original_image, image)
        elif str(probabilities[1]) in images_paths[i]:
            mean_psnr[1] += cv2.PSNR(original_image, image)
        elif str(probabilities[2]) in images_paths[i]:
            mean_psnr[2] += cv2.PSNR(original_image, image)
        elif str(probabilities[3]) in images_paths[i]:
            mean_psnr[3] += cv2.PSNR(original_image, image)
        elif str(probabilities[4]) in images_paths[i]:
            mean_psnr[4] += cv2.PSNR(original_image, image)
    
    for i, prob in enumerate(probabilities):
        print(f"Mean PSNR for {prob} = {mean_psnr[i] / num_images}")


'''
Test Mean Blur on the noisy images
'''
def test_mean_blur(images, images_paths, original_image, num_images, probabilities):
    print("\n==================================")
    print("Testing Mean Blur")

    k_size = [i for i in range(1, 10, 2)]

    mean_psnr = [0] * len(probabilities) * len(k_size)

    for i, image in enumerate(images):
        for j, size in enumerate(k_size):
            if str(probabilities[0]) in images_paths[i]:
                mean_psnr[0 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.blur(image, (size, size)))
            elif str(probabilities[1]) in images_paths[i]:
                mean_psnr[1 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.blur(image, (size, size)))
            elif str(probabilities[2]) in images_paths[i]:
                mean_psnr[2 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.blur(image, (size, size)))
            elif str(probabilities[3]) in images_paths[i]:
                mean_psnr[3 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.blur(image, (size, size)))
            elif str(probabilities[4]) in images_paths[i]:
                mean_psnr[4 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.blur(image, (size, size)))

    for i in range(len(mean_psnr)):
        mean_psnr[i] /= num_images

    results = []

    for i in range(len(probabilities)):
        print(f"Probabilitie = {probabilities[i]}")
        for j in range(len(k_size)):
            print(f"Mean PSNR for {k_size[j]} = {mean_psnr[i * len(probabilities) + j]}")

        results.append(mean_psnr[i * len(probabilities) + 1]) # USANDO KERNEL = 3
    
    return results


'''
Test Median Blur on the noisy images
'''
def test_median_blur(images, images_paths, original_image, num_images, probabilities):
    print("\n==================================")
    print("Testing Median Blur")

    k_size = [i for i in range(1, 10, 2)]

    mean_psnr = [0] * len(probabilities) * len(k_size)

    for i, image in enumerate(images):
        for j, size in enumerate(k_size):
            if str(probabilities[0]) in images_paths[i]:
                mean_psnr[0 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.medianBlur(image, size))
            elif str(probabilities[1]) in images_paths[i]:
                mean_psnr[1 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.medianBlur(image, size))
            elif str(probabilities[2]) in images_paths[i]:
                mean_psnr[2 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.medianBlur(image, size))
            elif str(probabilities[3]) in images_paths[i]:
                mean_psnr[3 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.medianBlur(image, size))
            elif str(probabilities[4]) in images_paths[i]:
                mean_psnr[4 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.medianBlur(image, size))

    for i in range(len(mean_psnr)):
        mean_psnr[i] /= num_images

    results = []

    for i in range(len(probabilities)):
        print(f"Probabilitie = {probabilities[i]}")
        for j in range(len(k_size)):
            print(f"Mean PSNR for {k_size[j]} = {mean_psnr[i * len(probabilities) + j]}")

        results.append(mean_psnr[i * len(probabilities) + 1]) # USANDO KERNEL = 3

    return results


'''
Test Gaussian Blur on the noisy images
'''
def test_gaussian_blur(images, images_paths, original_image, num_images, probabilities):
    print("\n==================================")
    print("Testing Gaussian Blur")

    k_size = [(i, i) for i in range(1, 10, 2)]
    sigma = 0

    mean_psnr = [0] * len(probabilities) * len(k_size)

    for i, image in enumerate(images):
        for j, size in enumerate(k_size):
            if str(probabilities[0]) in images_paths[i]:
                mean_psnr[0 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.GaussianBlur(image, size, sigma))
            elif str(probabilities[1]) in images_paths[i]:
                mean_psnr[1 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.GaussianBlur(image, size, sigma))
            elif str(probabilities[2]) in images_paths[i]:
                mean_psnr[2 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.GaussianBlur(image, size, sigma))
            elif str(probabilities[3]) in images_paths[i]:
                mean_psnr[3 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.GaussianBlur(image, size, sigma))
            elif str(probabilities[4]) in images_paths[i]:
                mean_psnr[4 * len(probabilities) + j] += cv2.PSNR(original_image, cv2.GaussianBlur(image, size, sigma))

    for i in range(len(mean_psnr)):
        mean_psnr[i] /= num_images

    results = []

    for i in range(len(probabilities)):
        print(f"Probabilitie = {probabilities[i]}")
        for j in range(len(k_size)):
            print(f"Mean PSNR for {k_size[j]} = {mean_psnr[i * len(probabilities) + j]}")

        results.append(mean_psnr[i * len(probabilities) + 1]) # USANDO KERNEL = 3
    
    return results


'''
Test Stacking on the noisy images
'''
def test_stacking(images, images_paths, original_image, num_images, probabilities, num_stacks):
    print("\n==================================")
    print("Testing Stacking")

    images_groups = [images[i:i + num_images] for i in range(0, len(images), num_images)]

    results = []

    for stack in range(10, num_stacks + 1, 10):
        print(f"\nStack = {stack}")

        for group_idx in range(len(images_groups)):

            image_list = [images_groups[group_idx][i] for i in range(stack)]
            image = np.mean(image_list, axis=0).astype(np.uint8)

            print(f"Probabilitie = {probabilities[group_idx]}")
            print(f"Mean PSNR = {cv2.PSNR(original_image, image)}")

            if stack == num_stacks:
                results.append(cv2.PSNR(original_image, image))
        
    return results


'''
Test Bilateral on the noisy images
'''
def test_bilateral(images, images_paths, original_image, num_images, probabilities):
    print("\n==================================")
    print("Testing Bilateral")

    results = []

    mean_psnr = [0] * len(probabilities)

    for i, image in enumerate(images):
        if str(probabilities[0]) in images_paths[i]:
            mean_psnr[0] += cv2.PSNR(original_image, cv2.bilateralFilter(image, 9, 75, 75))
        elif str(probabilities[1]) in images_paths[i]:
            mean_psnr[1] += cv2.PSNR(original_image, cv2.bilateralFilter(image, 9, 75, 75))
        elif str(probabilities[2]) in images_paths[i]:
            mean_psnr[2] += cv2.PSNR(original_image, cv2.bilateralFilter(image, 9, 75, 75))
        elif str(probabilities[3]) in images_paths[i]:
            mean_psnr[3] += cv2.PSNR(original_image, cv2.bilateralFilter(image, 9, 75, 75))
        elif str(probabilities[4]) in images_paths[i]:
            mean_psnr[4] += cv2.PSNR(original_image, cv2.bilateralFilter(image, 9, 75, 75))

    for i in range(len(mean_psnr)):
        mean_psnr[i] /= num_images
    
    for i in range(len(probabilities)):
        print(f"Probabilitie = {probabilities[i]}")
        print(f"Mean PSNR = {mean_psnr[i]}")

        results.append(mean_psnr[i])

    return results


'''
Test NLMeans on the noisy images
'''
def test_nlmeans(images, images_paths, original_image, num_images, probabilities):
    print("\n==================================")
    print("Testing NLMeans")

    results = []

    mean_psnr = [0] * len(probabilities)

    for i, image in enumerate(images):
        if str(probabilities[0]) in images_paths[i]:
            mean_psnr[0] += cv2.PSNR(original_image, cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21))
        elif str(probabilities[1]) in images_paths[i]:
            mean_psnr[1] += cv2.PSNR(original_image, cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21))
        elif str(probabilities[2]) in images_paths[i]:
            mean_psnr[2] += cv2.PSNR(original_image, cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21))
        elif str(probabilities[3]) in images_paths[i]:
            mean_psnr[3] += cv2.PSNR(original_image, cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21))
        elif str(probabilities[4]) in images_paths[i]:
            mean_psnr[4] += cv2.PSNR(original_image, cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21))

    for i in range(len(mean_psnr)):
        mean_psnr[i] /= num_images
    
    for i in range(len(probabilities)):
        print(f"Probabilitie = {probabilities[i]}")
        print(f"Mean PSNR = {mean_psnr[i]}")

        results.append(mean_psnr[i])

    return results


def test_stacking_median(images, images_paths, original_image, num_images, probabilities, stack):
    print("\n==================================")
    print("Testing Stacking Median")

    images_groups = [images[i:i + num_images] for i in range(0, len(images), num_images)]

    results = []

    print(f"\nStack = {stack}")

    for group_idx in range(len(images_groups)):

        image_list = [images_groups[group_idx][i] for i in range(stack)]
        image = np.median(image_list, axis=0).astype(np.uint8)

        print(f"Probabilitie = {probabilities[group_idx]}")
        print(f"Mean PSNR = {cv2.PSNR(original_image, image)}")

        results.append(cv2.PSNR(original_image, image))
        
    return results


original_image_path = sys.argv[1]
original_image = cv2.imread(original_image_path)

probabilities = [0.01, 0.02, 0.05, 0.07, 0.1]
num_images = 20
num_stacks = num_images

if not os.path.exists("images"):
    generate_noise_images(original_image, probabilities, num_images)

noise_images_paths = sorted(glob("images/**/*.png"))
noise_images = [cv2.imread(image) for image in noise_images_paths]

print_psnr(noise_images, noise_images_paths, original_image, num_images, probabilities)

mean_blur_results = test_mean_blur(noise_images, noise_images_paths, original_image, num_images, probabilities)
median_blur_results = test_median_blur(noise_images, noise_images_paths, original_image, num_images, probabilities)
gaussian_blur_results = test_gaussian_blur(noise_images, noise_images_paths, original_image, num_images, probabilities)
stacking_results = test_stacking(noise_images, noise_images_paths, original_image, num_images, probabilities, num_stacks)
bilateral_results = test_bilateral(noise_images, noise_images_paths, original_image, num_images, probabilities)
nlmeans_results = test_nlmeans(noise_images, noise_images_paths, original_image, num_images, probabilities)
stacking_median_results = test_stacking_median(noise_images, noise_images_paths, original_image, num_images, probabilities, stack=17)

print("\n==================================")

print(f"{'Filter x Noise':<15}{probabilities[0]:<15}{probabilities[1]:<15}{probabilities[2]:<15}{probabilities[3]:<15}{probabilities[4]:<15}")
print(f"{'Mean Blur':<15}{mean_blur_results[0]:<15.2f}{mean_blur_results[1]:<15.2f}{mean_blur_results[2]:<15.2f}{mean_blur_results[3]:<15.2f}{mean_blur_results[4]:<15.2f}")
print(f"{'Median Blur':<15}{median_blur_results[0]:<15.2f}{median_blur_results[1]:<15.2f}{median_blur_results[2]:<15.2f}{median_blur_results[3]:<15.2f}{median_blur_results[4]:<15.2f}")
print(f"{'Gaussian Blur':<15}{gaussian_blur_results[0]:<15.2f}{gaussian_blur_results[1]:<15.2f}{gaussian_blur_results[2]:<15.2f}{gaussian_blur_results[3]:<15.2f}{gaussian_blur_results[4]:<15.2f}")
print(f"{'Stacking':<15}{stacking_results[0]:<15.2f}{stacking_results[1]:<15.2f}{stacking_results[2]:<15.2f}{stacking_results[3]:<15.2f}{stacking_results[4]:<15.2f}")
print(f"{'Bilateral':<15}{bilateral_results[0]:<15.2f}{bilateral_results[1]:<15.2f}{bilateral_results[2]:<15.2f}{bilateral_results[3]:<15.2f}{bilateral_results[4]:<15.2f}")
print(f"{'NLMeans':<15}{nlmeans_results[0]:<15.2f}{nlmeans_results[1]:<15.2f}{nlmeans_results[2]:<15.2f}{nlmeans_results[3]:<15.2f}{nlmeans_results[4]:<15.2f}")
print(f"{'Stacking Median':<15}{stacking_median_results[0]:<15.2f}{stacking_median_results[1]:<15.2f}{stacking_median_results[2]:<15.2f}{stacking_median_results[3]:<15.2f}{stacking_median_results[4]:<15.2f}")
