from glob import glob
import numpy as np
import sys
import cv2
import os

input = sys.argv[1]

gerar_imagens = False
if len(sys.argv) == 3:
    output = sys.argv[2]
    gerar_imagens = True
    os.makedirs(output, exist_ok=True)

ascii_file = "results.txt"

images_paths = glob(f"{input}/*.png")
images = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in images_paths]
images = [cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1] for image in images]


# Determining the form type
form_types = []
for i, image in enumerate(images):
    area = image[0:200, 900:1500]

    area = cv2.bitwise_not(area)

    area = cv2.erode(area, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    area = cv2.dilate(area, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=10)

    area_count = cv2.countNonZero(area)

    if area_count > 60000:
        form_types.append(0)
    else:
        form_types.append(1)


first_time_yes_images = []
first_time_no_images = []
consider_services_yes_images = []
consider_services_no_images = []
recommend_yes_images = []
recommend_no_images = []
rating_images = []
excellent_images = []
good_images = []
fair_images = []
poor_images = []

first_time_yes_responses = []
first_time_no_responses = []
consider_services_yes_responses = []
consider_services_no_responses = []
recommend_yes_responses = []
recommend_no_responses = []
rating_responses = []
excellent_responses = []
good_responses = []
fair_responses = []
poor_responses = []


# Segmenting the images
intervals = [(680, 860), (860, 1040), (1040, 1220), (1220, 1400), (1400, 1580), (1580, 1760)]
for i, image in enumerate(images):
    excellent = []
    good = []
    fair = []
    poor = []

    area = image[1760:1940, 0:(image.shape[1] // 2) + 300]

    first_time_yes = area[0:180, 850:1200]
    first_time_no = area[0:180, 1200:1600]

    first_time_yes_images.append(first_time_yes)
    first_time_no_images.append(first_time_no)

    area = image[2100:2260, 0:image.shape[1] // 2]

    consider_services_yes = area[0:180, 0:400]
    consider_services_no = area[0:180, 400:800]

    consider_services_yes_images.append(consider_services_yes)
    consider_services_no_images.append(consider_services_no)

    area = image[2110:2260, (image.shape[1] // 2) + 100:]

    recommend_yes = area[0:180, 0:300]
    recommend_no = area[0:180, 300:700]

    recommend_yes_images.append(recommend_yes)
    recommend_no_images.append(recommend_no)
    
    rating = image[2300:2500, 900:(image.shape[1] - 100)]

    rating_images.append(rating)

    for interval in intervals:
        area = image[interval[0]:interval[1], 850:2479]

        height, width  = area.shape
        space = width // 4

        areas = []
        for j in range(4):
            areas.append(area[0:height, j*space:(j+1)*space])
        
        excellent.append(areas[0])
        good.append(areas[1])
        fair.append(areas[2])
        poor.append(areas[3])

    excellent_images.append(excellent)
    good_images.append(good)
    fair_images.append(fair)
    poor_images.append(poor)


# Applyng method on segmented images
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

type0_thresholds = [6225, 4200, 3100, 3200]
type1_thresholds = [8500, 5500, 4600, 4800]
first_time0_thresholds = [3200, 3100]
first_time1_thresholds = [4100, 3600]
consider0_thresholds = [3200, 2900]
consider1_thresholds = [4600, 3900]
recommend0_thresholds = [3300, 3000]
recommend1_thresholds = [4100, 3700]

for i, image in enumerate(images):
    form_type = form_types[i]

    excellent_responses.append([])
    good_responses.append([])
    fair_responses.append([])
    poor_responses.append([])

    for responses in range(6):
        responses_images = []
        responses_images.append(excellent_images[i][responses])
        responses_images.append(good_images[i][responses])
        responses_images.append(fair_images[i][responses])
        responses_images.append(poor_images[i][responses])
        responses_images = [cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1] for image in responses_images]
        responses_images = [cv2.bitwise_not(image) for image in responses_images]
        responses_images = [cv2.erode(image, kernel1, iterations=1) for image in responses_images]
        responses_images = [cv2.dilate(image, kernel2, iterations=1) for image in responses_images]

        areas = [cv2.countNonZero(image) for image in responses_images]

        for j in range(len(areas)):
            if form_type == 0:
                if areas[j] > type0_thresholds[j]:
                    if j == 0:
                        excellent_responses[i].append(1)
                    elif j == 1:
                        good_responses[i].append(1)
                    elif j == 2:
                        fair_responses[i].append(1)
                    elif j == 3:
                        poor_responses[i].append(1)
                else:
                    if j == 0:
                        excellent_responses[i].append(0)
                    elif j == 1:
                        good_responses[i].append(0)
                    elif j == 2:
                        fair_responses[i].append(0)
                    elif j == 3:
                        poor_responses[i].append(0)
                    
            else:
                if areas[j] > type1_thresholds[j]:
                    if j == 0:
                        excellent_responses[i].append(1)
                    elif j == 1:
                        good_responses[i].append(1)
                    elif j == 2:
                        fair_responses[i].append(1)
                    elif j == 3:
                        poor_responses[i].append(1)
                else:
                    if j == 0:
                        excellent_responses[i].append(0)
                    elif j == 1:
                        good_responses[i].append(0)
                    elif j == 2:
                        fair_responses[i].append(0)
                    elif j == 3:
                        poor_responses[i].append(0)

    first_time_image = []
    first_time_image.append(first_time_yes_images[i])
    first_time_image.append(first_time_no_images[i])
    first_time_image = [cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1] for image in first_time_image]
    first_time_image = [cv2.bitwise_not(image) for image in first_time_image]
    first_time_image = [cv2.erode(image, kernel1, iterations=1) for image in first_time_image]
    first_time_image = [cv2.dilate(image, kernel2, iterations=1) for image in first_time_image]

    areas = [cv2.countNonZero(image) for image in first_time_image]

    if form_type == 0:
        if areas[0] > first_time0_thresholds[0]:
            first_time_yes_responses.append(1)
        else:
            first_time_yes_responses.append(0)
        
        if areas[1] > first_time0_thresholds[1]:
            first_time_no_responses.append(1)
        else:
            first_time_no_responses.append(0)
    
    else:
        if areas[0] > first_time1_thresholds[0]:
            first_time_yes_responses.append(1)
        else:
            first_time_yes_responses.append(0)
        if areas[1] > first_time1_thresholds[1]:
            first_time_no_responses.append(1)
        else:
            first_time_no_responses.append(0)

    consider_image = []
    consider_image.append(consider_services_yes_images[i])
    consider_image.append(consider_services_no_images[i])
    consider_image = [cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1] for image in consider_image]
    consider_image = [cv2.bitwise_not(image) for image in consider_image]
    consider_image = [cv2.erode(image, kernel1, iterations=1) for image in consider_image]
    consider_image = [cv2.dilate(image, kernel2, iterations=1) for image in consider_image]

    areas = [cv2.countNonZero(image) for image in consider_image]

    if form_type == 0:
        if areas[0] > consider0_thresholds[0]:
            consider_services_yes_responses.append(1)
        else:
            consider_services_yes_responses.append(0)
        
        if areas[1] > consider0_thresholds[1]:
            consider_services_no_responses.append(1)
        else:
            consider_services_no_responses.append(0)
    else:
        if areas[0] > consider1_thresholds[0]:
            consider_services_yes_responses.append(1)
        else:
            consider_services_yes_responses.append(0)
        
        if areas[1] > consider1_thresholds[1]:
            consider_services_no_responses.append(1)
        else:
            consider_services_no_responses.append(0)

    recommend_image = []
    recommend_image.append(recommend_yes_images[i])
    recommend_image.append(recommend_no_images[i])
    recommend_image = [cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1] for image in recommend_image]
    recommend_image = [cv2.bitwise_not(image) for image in recommend_image]
    recommend_image = [cv2.erode(image, kernel1, iterations=1) for image in recommend_image]
    recommend_image = [cv2.dilate(image, kernel2, iterations=1) for image in recommend_image]

    areas = [cv2.countNonZero(image) for image in recommend_image]

    if form_type == 0:
        if areas[0] > recommend0_thresholds[0]:
            recommend_yes_responses.append(1)
        else:
            recommend_yes_responses.append(0)
        
        if areas[1] > recommend0_thresholds[1]:
            recommend_no_responses.append(1)
        else:
            recommend_no_responses.append(0)
    else:
        if areas[0] > recommend1_thresholds[0]:
            recommend_yes_responses.append(1)
        else:
            recommend_yes_responses.append(0)
        
        if areas[1] > recommend1_thresholds[1]:
            recommend_no_responses.append(1)
        else:
            recommend_no_responses.append(0)

    rating_image = rating_images[i]
    rating_image = cv2.threshold(rating_image, 127, 255, cv2.THRESH_BINARY)[1]
    rating_image = cv2.bitwise_not(rating_image)
    rating_image = cv2.erode(rating_image, kernel1, iterations=1)
    rating_image = cv2.dilate(rating_image, kernel2, iterations=5)

    rating_range = rating_image.shape[1] // 10
    rating_images_ranged = [rating_image[:, i*rating_range:(i+1)*rating_range] for i in range(10)]
    
    areas = [cv2.countNonZero(image) for image in rating_images_ranged]

    indice = areas.index(max(areas)) + 1

    rating_responses.append(indice)


# Drawing
if gerar_imagens:
    normal_images = [cv2.imread(image) for image in images_paths]
    for i, image in enumerate(normal_images):
        if first_time_yes_responses[i] == 1:
            cv2.line(image, (900, 1940), ((image.shape[1] // 2) - 200, 1940), (0, 0, 255), 5)
        
        if first_time_no_responses[i] == 1:
            cv2.line(image, ((image.shape[1] // 2), 1940), ((image.shape[1] // 2) + 200, 1940), (0, 0, 255), 5)

        if consider_services_yes_responses[i] == 1:
            cv2.line(image, (100, 2260), (300, 2260), (0, 0, 255), 5)
        
        if consider_services_no_responses[i] == 1:
            cv2.line(image, (400, 2260), (600, 2260), (0, 0, 255), 5)

        if recommend_yes_responses[i] == 1:
            cv2.line(image, ((image.shape[1] // 2) + 200, 2260), ((image.shape[1] // 2) + 400, 2260), (0, 0, 255), 5)
        
        if recommend_no_responses[i] == 1:
            cv2.line(image, ((image.shape[1] // 2) + 500, 2260), ((image.shape[1] // 2) + 700, 2260), (0, 0, 255), 5)

        cv2.line(image, ((900 + (rating_responses[i] - 1) * rating_range), 2500), ((900 + rating_responses[i] * rating_range), 2500), (0, 0, 255), 5)


        width = 1630
        space = width // 4

        for j in range(6):
            if excellent_responses[i][j] == 1:
                cv2.line(image, (850, intervals[j][1]), (850 + space, intervals[j][1]), (0, 0, 255), 5)
            if good_responses[i][j] == 1:
                cv2.line(image, (850 + space, intervals[j][1]), (850 + 2*space, intervals[j][1]), (0, 0, 255), 5)
            if fair_responses[i][j] == 1:
                cv2.line(image, (850 + 2*space, intervals[j][1]), (850 + 3*space, intervals[j][1]), (0, 0, 255), 5)
            if poor_responses[i][j] == 1:
                cv2.line(image, (850 + 3*space, intervals[j][1]), (850 + 4*space, intervals[j][1]), (0, 0, 255), 5)

        image_name = (images_paths[i].split("/")[-1]).split(".")[0]
        if output.endswith("/"):
            cv2.imwrite(f"{output}{image_name}", image)
        else:
            cv2.imwrite(f"{output}/{image_name}.out.png", image)


# Counting responses
line_1 = [0] * 4
line_2 = [0] * 4
line_3 = [0] * 4
line_4 = [0] * 4
line_5 = [0] * 4
line_6 = [0] * 4
lines = [line_1, line_2, line_3, line_4, line_5, line_6]

for i in range(len(images_paths)):
    for j in range(6):
        if excellent_responses[i][j] == 1:
            lines[j][0] += 1
        if good_responses[i][j] == 1:
            lines[j][1] += 1
        if fair_responses[i][j] == 1:
            lines[j][2] += 1
        if poor_responses[i][j] == 1:
            lines[j][3] += 1

first_time_yes_count = 0
first_time_no_count = 0
for i in range(len(images_paths)):
    if first_time_yes_responses[i] == 1:
        first_time_yes_count += 1
    if first_time_no_responses[i] == 1:
        first_time_no_count += 1

consider_services_yes_count = 0
consider_services_no_count = 0
for i in range(len(images_paths)):
    if consider_services_yes_responses[i] == 1:
        consider_services_yes_count += 1
    if consider_services_no_responses[i] == 1:
        consider_services_no_count += 1

recommend_yes_count = 0
recommend_no_count = 0
for i in range(len(images_paths)):
    if recommend_yes_responses[i] == 1:
        recommend_yes_count += 1
    if recommend_no_responses[i] == 1:
        recommend_no_count += 1

mean_rating = np.mean(rating_responses) * 10

with open(ascii_file, "w") as file:
    file.write(f"{line_1[0] / sum(line_1) * 100:.2f} {line_1[1] / sum(line_1) * 100:.2f} {line_1[2] / sum(line_1) * 100:.2f} {line_1[3] / sum(line_1) * 100:.2f}\n")
    file.write(f"{line_2[0] / sum(line_2) * 100:.2f} {line_2[1] / sum(line_2) * 100:.2f} {line_2[2] / sum(line_2) * 100:.2f} {line_2[3] / sum(line_2) * 100:.2f}\n")
    file.write(f"{line_3[0] / sum(line_3) * 100:.2f} {line_3[1] / sum(line_3) * 100:.2f} {line_3[2] / sum(line_3) * 100:.2f} {line_3[3] / sum(line_3) * 100:.2f}\n")
    file.write(f"{line_4[0] / sum(line_4) * 100:.2f} {line_4[1] / sum(line_4) * 100:.2f} {line_4[2] / sum(line_4) * 100:.2f} {line_4[3] / sum(line_4) * 100:.2f}\n")
    file.write(f"{line_5[0] / sum(line_5) * 100:.2f} {line_5[1] / sum(line_5) * 100:.2f} {line_5[2] / sum(line_5) * 100:.2f} {line_5[3] / sum(line_5) * 100:.2f}\n")
    file.write(f"{line_6[0] / sum(line_6) * 100:.2f} {line_6[1] / sum(line_6) * 100:.2f} {line_6[2] / sum(line_6) * 100:.2f} {line_6[3] / sum(line_6) * 100:.2f}\n")

    file.write(f"{first_time_yes_count / len(images_paths) * 100:.2f} {first_time_no_count / len(images_paths) * 100:.2f}\n")
    file.write(f"{consider_services_yes_count / len(images_paths) * 100:.2f} {consider_services_no_count / len(images_paths) * 100:.2f}\n")
    file.write(f"{recommend_yes_count / len(images_paths) * 100:.2f} {recommend_no_count / len(images_paths) * 100:.2f}\n")
    file.write(f"{mean_rating}\n")