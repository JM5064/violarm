import cv2
import csv
import os

image_folder = 'images_unlabeled'
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
result_folder = 'images_labeled'
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
            save_points()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # label as not visible
        cv2.putText(image, f"{keypoints[len(points)]}: ({x},{y})", (10, 20 + 25 * len(points)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow("Image", image)

        points.append([-1, -1])

        if len(points) == 8:
            save_points()


# save points to csv and move image to labeled folder
def save_points():
    global points, image_path

    image_name = image_path.removeprefix(image_folder)
    result_path = result_folder + image_name

    os.rename(image_path, result_path)

    with open(csv_file, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([result_path] + [coord for point in points for coord in point])

    print(f"Points saved to {csv_file} for {image_name}: {points}")
    
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


if image_paths:
    image_path = image_paths.pop(0)
    image = cv2.imread(image_path)

    cv2.imshow("Image", image)

    cv2.setMouseCallback("Image", click_event)

    cv2.waitKey(0)

print("done ")
cv2.destroyAllWindows()
