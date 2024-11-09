import cv2
import random

def display_image(image, window_message):
    cv2.imshow(window_message, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows


def draw_matches(grayscale_image1, grayscale_image2, keypoints1, keypoints2, matches):
        matched_image = cv2.drawMatches(grayscale_image1, keypoints1, grayscale_image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        display_image(matched_image, "Matches")


def display_contours(image, contours):
        image_copy = image.copy()

        for contour in contours:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(image_copy, [contour], -1, color, 2)

        display_image(image_copy, "Contours")

        return image_copy