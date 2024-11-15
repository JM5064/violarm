import cv2
import numpy as np
import random

def display_image(image, window_message):
    cv2.imshow(window_message, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows


def draw_matches(grayscale_image1, grayscale_image2, keypoints1, keypoints2, matches):
        matched_image = cv2.drawMatches(grayscale_image1, keypoints1, grayscale_image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        display_image(matched_image, "Matches")


def display_contours(image, contours, shift_x, shift_y):
        image_copy = image.copy()

        shifted_contours = []
        for contour in contours:
            shifted_contour = contour + np.array([shift_x, shift_y], dtype=np.int32)
            shifted_contours.append(shifted_contour)


        for contour in shifted_contours:
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            cv2.drawContours(image_copy, [contour], -1, color, 3)

        display_image(image_copy, "Contours")

        return image_copy