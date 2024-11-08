import cv2
import numpy as np
import scipy.signal
import random
import time


class Arm:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        still_image = self.get_still_image()
        grayscale_still_image = cv2.cvtColor(still_image, cv2.COLOR_RGB2GRAY)

        self.width = still_image.shape[1]
        self.height = still_image.shape[0]

        still_image_copy = still_image.copy()
        self.wrist_start, self.wrist_end = self.prompt_selection(still_image_copy, "Draw Selection Around Wrist")

        wrist_crop, _, _ = self.crop_image(grayscale_still_image, self.wrist_start, self.wrist_end, 0)

        self.wrist_keypoints, self.wrist_descriptors = self.get_keypoints(wrist_crop)

        print("Initialized Arm Model")



    def get_still_image(self):
        # press q when ready to draw
        while True:
            ret, frame = self.cap.read()

            cv2.imshow("Press 'q' when ready", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return frame


    def prompt_selection(self, frame, message):
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


    def get_matches(self, descriptors1, descriptors2, threshold):
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
    

    def get_keypoints(self, grayscale_image):
        """Get the keypoints and descriptors of an image
        args:
            grayscale_image
        
        returns:
            keypoints and descriptors of the image
        """

        detector = cv2.ORB_create()
        keypoints, descriptors = detector.detectAndCompute(grayscale_image, None)

        return keypoints, descriptors


    def crop_image(self, image, top_left_point, bottom_right_point, padding):
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
        if x2 > self.width:
            x2 = self.width
        if y2 > self.height:
            y2 = self.height

        return image[y1:y2, x1:x2], x1, y1


    def find_center(self, matches, keypoints2, distance_threshold):
        """Computes the center of the matched points
        args: 
            matches: matches from image1 to image2
            keypoints2: keypoints from the matched image
            distance_threshold: metric to deduce inliers
        
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
            if (distance < distance_threshold):
                refined_points.append(point)

        if len(refined_points) == 0:
            refined_center = center
        else:
            refined_center = np.mean(refined_points, axis=0)

        return round(refined_center[0]), round(refined_center[1])
        

    def filter_small_contours(self, grayscale_image, size_threshold):
        """Filters out the small contours
        args:
            grayscale_image: cropped grayscale image
            size_threshold: threshold for min contour arcLength
        
        returns:
            large_contours: list
        """

        edges = cv2.Canny(grayscale_image, 50, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # threshold only large contours
        large_contours = []
        for contour in contours:
            if cv2.arcLength(contour, False) > size_threshold:
                large_contours.append(contour)

        return large_contours


    def find_corners(self, grayscale_image, center, contours):
        """Finds the top corners of the bounding contours around center pixel
        args:
            grayscale_image: cropped grayscale image
            center: (int, int) center pixel
            contours: large contours to examine
        
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


        w = grayscale_image.shape[1]
        h = grayscale_image.shape[0]

        # find top left corner
        x1, y1 = center
        while x1 - 1 > -1 and y1 - 1 > -1 and contour_image[y1][x1] == 0:
            x1 -= 1
            y1 -= 1

        max_value = 0
        max_x1, max_y1 = x1, y1
        while x1 - 1 > -1 and y1 - 1 > -1:
            if x1 - 2 > -1 and y1 - 2 > -1 and x1 + 2 < w and y1 + 2 < h:
                cropped = contour_image[y1-2:y1+3, x1-2:x1+3]
                convolved = scipy.signal.convolve(laplacian, cropped)

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

        # find top right corner
        x2, y2 = center
        while x2 + 1 < w and y2 - 1 > -1 and contour_image[y2][x2] == 0:
            x2 += 1
            y2 -= 1

        max_value = 0
        max_x2, max_y2 = x2, y2
        while x2 + 1 < w and y2 - 1 > 0:
            if x2 - 2 > -1 and y2 - 2 > -1 and x2 + 2 < w and y2 + 2 < h:
                cropped = contour_image[y2-2:y2+3, x2-2:x2+3]
                convolved = scipy.signal.convolve(laplacian, cropped)

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

    
    def calculate_new_start_end(self, center, padding):
        """Calculates the new start and end points for a crop based on the center of the current image
        args:
            center: (int, int) point representing the old center
            padding: expanded area from center

        returns:
            start_point, end_point: (int, int), (int, int)
        """

        x, y = center

        start_point = (max(x - padding, 0), max(y - padding, 0))
        end_point = (min(x + padding, self.width - 1), min(y + padding, self.height - 1))

        return start_point, end_point


# still_image_copy = still_image.copy()
# wrist_start, wrist_end = prompt_selection(still_image_copy, "Draw Selection Around Wrist")

# wrist_crop, _, _ = crop_image(grayscale_still_image, wrist_start, wrist_end, 0)


# # get second image
# still_image2 = get_still_image()
# grayscale_still_image2 = cv2.cvtColor(still_image2, cv2.COLOR_RGB2GRAY)

# # crop from second image
# padding = 200
# wrist_crop_larger, cropped_x, cropped_y = crop_image(grayscale_still_image2, wrist_start, wrist_end, padding)

# # match first crop to second crop
# keypoints1, descriptors1 = get_keypoints(wrist_crop)
# keypoints2, descriptors2 = get_keypoints(wrist_crop_larger)

# matches = get_matches(descriptors1, descriptors2, 300)

# draw_matches(wrist_crop, wrist_crop_larger, keypoints1, keypoints2, matches)
# center_x, center_y = find_center(matches, keypoints2, 50)
# # center_x += cropped_x1
# # center_y += cropped_y1

# # centerimage = cv2.circle(still_image2, (center_x, center_y), 5, (0, 0, 255), 10)
# # display_image(centerimage, "Center")

# large_contours = filter_small_contours(wrist_crop_larger, 300)
# x1, y1, x2, y2 = find_corners(wrist_crop_larger, (center_x, center_y), large_contours)
# x1 += cropped_x
# x2 += cropped_x
# y1 += cropped_y
# y2 += cropped_y

# corner_image = cv2.circle(still_image2, (x1, y1), 3, (0, 0, 255), 5)
# cv2.circle(still_image2, (x2, y2), 3, (0, 0, 255), 5)
# display_image(corner_image, "Corners")


# cap.release()
# cv2.destroyAllWindows()
