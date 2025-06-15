import cv2
import csv
import os

image_folder = 'images_unlabeled'
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
result_folder = 'images_labeled'
skip_folder = 'images_skipped'
csv_file = 'data.csv'

keypoints = {
    0: "first_finger",
    1: "second_finger",
    2: "third_finger",
    3: "fourth_finger",
    4: "arm_top_left",
    5: "arm_top_right",
    6: "arm_bottom_right",
    7: "arm_bottom_left"
}

points = []

def click_event(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        # mark first four finger points with red, last four arm points with green
        if len(points) < 4:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.circle(image, (x, y), 5, color, -1)

        cv2.putText(image, f"{keypoints[len(points)]}: ({x},{y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow("Image", image)

        points.append([x, y])

        if len(points) == 8:
            if not correct_labels():
                points = []
            else:
                save_points()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # label as not visible
        cv2.putText(image, f"{keypoints[len(points)]}: ({x},{y})", (10, 20 + 25 * len(points)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow("Image", image)

        points.append([-1, -1])

        if len(points) == 8:
            if not correct_labels():
                points = []
            else:
                save_points()


# save points to csv and move image to labeled folder
def save_points():
    global points, image_path

    result_path = image_path.replace(image_folder, result_folder)

    os.rename(image_path, result_path)

    with open(csv_file, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([result_path] + [coord for point in points for coord in point])

    print(f"Points saved to {csv_file} for {result_path}: {points}")
    
    points = []

    next_image()


# load next image
def next_image():
    global image, image_path, image_paths

    if image_paths:
        image_path = image_paths.pop(0)
        image = cv2.imread(image_path)
        cv2.imshow("Image", image)
    else:
        print("All images have been processed")
        cv2.destroyAllWindows()


def skip_image():
    global points
    points = []

    skip_path = image_path.replace(image_folder, skip_folder)
    os.rename(image_path, skip_path)


def correct_labels():
    global image_path

    # check finger labels
    # if not (points[0][1] <= points[1][1] <= points[2][1] <= points[3][1]):
    #     print(f"{image_path} Labeled fingers incorrectly")
    #     return False
    
    # check arm labels
    if not (points[4][0] <= points[5][0] and points[5][1] <= points[6][1] and 
            points[6][0] >= points[7][0] and points[7][1] >= points[4][1]):
        print(f"{image_path} Labeled arm incorrectly")
        return False

    return True


while image_paths:
    image_path = image_paths.pop(0)
    image = cv2.imread(image_path)

    cv2.imshow("Image", image)

    cv2.setMouseCallback("Image", click_event)

    # press space to skip image. otherwise, quit
    key = cv2.waitKey(0)
    if key == 32:
        skip_image()
    else:
        break

print("done")
cv2.destroyAllWindows()
