import cv2
import mediapipe as mp
import numpy as np
import scipy.signal
import random
import time


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def get_still_image():
    # press q when ready to draw
    while True:
        ret, frame = cap.read()

        cv2.imshow("Press 'q' when ready", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return frame


still_image = get_still_image()
width = still_image.shape[1]
height = still_image.shape[0]


def prompt_selection(frame, message):
    initial_frame_copy = frame.copy()

    drawing = False
    finished = False

    start_point = None
    end_point = None

    def draw_rectangle(event, x, y, flags, param):
        nonlocal frame, initial_frame_copy, start_point, end_point, drawing, finished
        
        # undo drawing
        if event == cv2.EVENT_RBUTTONDOWN:
            finished = False
            start_point = None
            end_point = None
            
            frame = initial_frame_copy
            initial_frame_copy = frame.copy()

        if finished:
            return

        # draw rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            finished = True
            

    cv2.namedWindow(message)
    cv2.setMouseCallback(message, draw_rectangle)

    while True:
        temp_frame = frame.copy()
        if drawing and end_point is not None:
            cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow(message, temp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') and start_point is not None and end_point is not None:
            if start_point[0] < end_point[0] and start_point[1] < end_point[1]:
                # TODO
                # only allowed to draw from top left to bottom right. add other case
                cv2.destroyAllWindows()
                
                return start_point, end_point
            

def display_contours(image, contours):
    # grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(grayscale_image, 50, 150)

    # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    image_copy = image.copy()

    for contour in contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(image_copy, [contour], -1, color, 2)

    display_image(image_copy, "Contours")

    return image_copy


def display_image(image, window_message):
    cv2.imshow(window_message, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows
            

def get_keypoints(grayscale_image):
    """Get the keypoints and descriptors of an image
    args:
        grayscale_image
    
    returns:
        keypoints and descriptors of the image
    """

    detector = cv2.ORB_create()
    keypoints, descriptors = detector.detectAndCompute(grayscale_image, None)

    return keypoints, descriptors


def get_matches(descriptors1, descriptors2, threshold):
    """Gets the matching keypoints of two images
    args:
        descriptors1, descriptors2: descriptors of the images
        threshold: int metric to keep best matches
    
    returns:
        list of matches
    """

    # match the keypoints of the two images
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors1, descriptors2)

    # threshold matches
    return [m for m in matches if m.distance < threshold]


def draw_matches(grayscale_image1, grayscale_image2, keypoints1, keypoints2, matches):
    matched_image = cv2.drawMatches(grayscale_image1, keypoints1, grayscale_image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    display_image(matched_image, "Matches")


def crop_image(image, top_left_point, bottom_right_point, padding):
    """Crops the image given specified points and padding
    args:
        image: input image to crop
        top_left_point: top left point of the crop
        bottom_right_point: bottom right point of the crop
        padding: adds padding to the extend the area cropped

    returns:
        cropped image
        x1, y1: coordinates of the top left corner of crop wrt the original image
    """

    x1, y1 = top_left_point
    x2, y2 = bottom_right_point

    x1 -= padding
    y1 -= padding

    x2 += padding
    y2 += padding

    # make sure x1, y1 are still in bounds
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > width:
        x2 = width
    if y2 > height:
        y2 = height

    return image[y1:y2, x1:x2], x1, y1


def find_center(matches, keypoints2, threshold):
    """Computes the center of the matched points
    args: 
        matches: matches from image1 to image2
        keypoints2: keypoints from the matched image
        threshold: metric to deduce inliers
    
    returns:
        x, y: int, int
    """
    
    # gather the matching points in the second image
    points = []
    for match in matches:
        points.append(keypoints2[match.trainIdx].pt)

    # calculate mean location of points
    center = np.mean(points, axis=0)

    refined_points = []
    # recalculate distances of each point to the center and keep the inliers
    for point in points:
        distance = np.linalg.norm(point - center)
        if (distance < threshold):
            refined_points.append(point)

    if len(refined_points) == 0:
        refined_center = center
    else:
        refined_center = np.mean(refined_points, axis=0)

    return round(refined_center[0]), round(refined_center[1])
    

def filter_small_contours(grayscale_image, size_threshold):
    """Filters out the small contours
    args:
        grayscale_image: cropped grayscale image
        size_threshold: threshold for min contour arcLength
    
    returns:
        large_contours: list
    """

    edges = cv2.Canny(grayscale_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # threshold only large contours
    large_contours = []
    for contour in contours:
        if cv2.arcLength(contour, False) > size_threshold:
            large_contours.append(contour)

    return large_contours


def find_corners(grayscale_image, center, contours):
    """Finds the top corners of the bounding contours around center pixel
    args:
        grayscale_image: cropped grayscale image
        center: (int, int) center pixel
        contours: large contours to exaime
    
    returns:    
        x1, y1: coordinate of top left corner
        x2, y2: coordinate of top right corner
    
    """
    
    # create image with only contours
    contour_image = np.zeros(grayscale_image.shape)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)


    laplacian = np.array([[1, 0, -2, 0, 1],
                         [0, -2, 0, 2, 0],
                         [-2, 0, 4, 0, -2],
                         [0, 2, 0, -2, 0],
                         [1, 0, -2, 0, 1]])


    # find top left corner
    x1, y1 = center
    w = grayscale_image.shape[1]
    h = grayscale_image.shape[0]


    # find edge
    while x1 > -1 and y1 > -1 and contour_image[y1][x1] == 0:
        x1 -= 1
        y1 -= 1

    max_value = 0
    max_x1, max_y1 = x1, y1
    while x1 > -1 and y1 > -1:
        if x1 - 2 > -1 and y1 - 2 > -1 and x1 + 2 < w and y1 + 2 < h:
            cropped = contour_image[y1-2:y1+3, x1-2:x1+3]
            convolved = scipy.signal.fftconvolve(laplacian, cropped)

            if abs(convolved[3][3]) > max_value:
                max_value = abs(convolved[3][3])
                max_x1, max_y1 = x1, y1

        if contour_image[y1-1][x1-1] == 255:
            x1 -= 1
            y1 -= 1
        elif contour_image[y1+1][x1-1] == 255:
            x1 -= 1
            y1 += 1
        elif contour_image[y1-1][x1] == 255:
            y1 -= 1
        elif contour_image[y1][x1-1] == 255:
            x1 -= 1
        else:
            break

    # ---------------------------------------
    print("---------------------------------------")

    # find top right corner
    x2, y2 = center
    while x2 < w and y2 > 0 and contour_image[y2][x2] == 0:
        x2 += 1
        y2 -= 1

    max_value = 0
    max_x2, max_y2 = x2, y2
    while x2 < w and y2 > 0:
        if x2 - 2 > -1 and y2 - 2 > -1 and x2 + 2 < w and y2 + 2 < h:
            cropped = contour_image[y2-2:y2+3, x2-2:x2+3]
            convolved = scipy.signal.fftconvolve(laplacian, cropped)

            if abs(convolved[3][3]) > max_value:
                max_value = abs(convolved[3][3])
                max_x2, max_y2 = x2, y2

        if contour_image[y2-1][x2+1] == 255:
            x2 += 1
            y2 -= 1
        elif contour_image[y2+1][x2+1] == 255:
            x2 += 1
            y2 += 1
        elif contour_image[y2-1][x2] == 255:
            y2 -= 1
        elif contour_image[y2][x2+1] == 255:
            x2 += 1
        else:
            break


    return max_x1, max_y1, max_x2, max_y2

    

grayscale_still_image = cv2.cvtColor(still_image, cv2.COLOR_RGB2GRAY)

still_image_copy = still_image.copy()
wrist_start, wrist_end = prompt_selection(still_image_copy, "Draw Selection Around Wrist")

wrist_crop, _, _ = crop_image(grayscale_still_image, wrist_start, wrist_end, 0)


# get second image
still_image2 = get_still_image()
grayscale_still_image2 = cv2.cvtColor(still_image2, cv2.COLOR_RGB2GRAY)

# crop from second image
padding = 200
wrist_crop_larger, cropped_x1, cropped_y1 = crop_image(grayscale_still_image2, wrist_start, wrist_end, padding)

# match first crop to second crop
keypoints1, descriptors1 = get_keypoints(wrist_crop)
keypoints2, descriptors2 = get_keypoints(wrist_crop_larger)

matches = get_matches(descriptors1, descriptors2, 300)

draw_matches(wrist_crop, wrist_crop_larger, keypoints1, keypoints2, matches)
center_x, center_y = find_center(matches, keypoints2, 50)
# center_x += cropped_x1
# center_y += cropped_y1

# centerimage = cv2.circle(still_image2, (center_x, center_y), 5, (0, 0, 255), 10)
# display_image(centerimage, "Center")

contour_image = find_corners(wrist_crop_larger, (center_x, center_y), 300)
display_image(contour_image, "Contour image")


cap.release()
cv2.destroyAllWindows()
