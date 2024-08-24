import sys
import cv2


def segment(input_image_path, output_image_path):
    image = cv2.imread(input_image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (35, 50, 50), (80, 255, 255))

    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    cv2.imwrite(output_image_path, segmented_image)


if __name__ == "__main__":
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    segment(input_image_path, output_image_path)
