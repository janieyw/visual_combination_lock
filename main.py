import cv2 as cv
import numpy as np
import math

def resize_img(img):  # resize to 300px X 300px
    height, width = img.shape[:2]
    scaling_factor = 300.0 / height
    dimensions = (int(width * scaling_factor), 300)
    return cv.resize(img, dimensions, interpolation = cv.INTER_AREA)

def convert_to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def convert_to_binary(img):
    gray_img = convert_to_gray(img)
    _, thresh_img = cv.threshold(gray_img, 160, 340, cv.THRESH_BINARY)
    thresh_img = cv.erode(thresh_img, None, iterations = 2)
    thresh_img = cv.dilate(thresh_img, None, iterations = 2)
    return thresh_img

def darken_non_red_regions(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_red = (0, 50, 50)
    upper_red = (10, 255, 255)
    mask = cv.inRange(hsv_img, lower_red, upper_red)
    img[~mask] = img[~mask] // 2
    return img

def detect_hand(img):
    # darken_non_red_regions(img)
    thresh_img = convert_to_binary(img)
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key = cv.contourArea)
    x, y, w, h = cv.boundingRect(max_contour)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow("Hand Detection", img)
    return x, y, w, h

# referenced Izane's Github (https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python/blob/master/new.py)
def count_fingers(masked_frame, drawing):
    hull = cv.convexHull(masked_frame, returnPoints = False)
    if len(hull) > 3:
        defects = cv.convexityDefects(masked_frame, hull)
        if defects is not None:
            fold_count = 0
            for i in range(defects.shape[0]):
                start_index, end_index, farthest_index, _ = defects[i][0]
                start = tuple(masked_frame[start_index][0])
                end = tuple(masked_frame[end_index][0])
                far = tuple(masked_frame[farthest_index][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # Cosine theorem
                if angle <= math.pi / 2:  # If angle less than 90 degree, treat as fingers
                    fold_count += 1
                    cv.circle(drawing, far, 8, [211, 84, 0], -1)
            return fold_count + 1  # Plus 1, as the count is for folds between fingers
    return 0

# referenced Izane's Github
def get_finger_count(img, x, y, w, h):
    thresh_img = convert_to_binary(img)
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    length = len(contours)

    max_contour_area = -1

    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv.contourArea(temp)
            if area > max_contour_area:
                max_contour_area = area
                index = i
        masked_frame = contours[index]
        hull = cv.convexHull(masked_frame)
        contour_area = cv.contourArea(masked_frame)
        hull_area = cv.contourArea(hull)
        area_diff = hull_area - contour_area
        drawing = np.zeros(img.shape, np.uint8)
        cv.drawContours(drawing, [masked_frame], 0, (0, 255, 0), 2)
        cv.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        finger_count = count_fingers(masked_frame, drawing)

    # cv.imshow('output', drawing)
    return finger_count, area_diff

def create_pose_label(img, x, y, w, h):
    finger_count, area_diff = get_finger_count(img, x, y, w, h)
    print(area_diff)
    if finger_count <= 1:
        pose = "fist"
    elif finger_count == 5:
        if area_diff > 3000:
            pose = "splay"
        else:
            pose = "palm"
    else:
        pose = "unknown"
    return pose

def create_location_label(img, x, y, w, h):
    # Get the length of the image
    img_length = img.shape[0]

    # Calculate the length of each 3x3 square within the image
    sq_length = img_length // 3

    # Identify the center of the bounding rectangle surrounding the hand
    center_x, center_y = x + w // 2, y + h // 2

    # Determine which of the 9 smaller squares the center of the bounding rectangle falls into
    if center_x < sq_length:
        if center_y < sq_length:
            location = "uppL"
        elif center_y < 2 * sq_length:
            location = "cenL"
        else:
            location = "lowL"
    elif center_x < 2 * sq_length:
        if center_y < sq_length:
            location = "lowC"
        elif center_y < 2 * sq_length:
            location = "cenC"
        else:
            location = "uppC"
    else:
        if center_y < sq_length:
            location = "uppR"
        elif center_y < 2 * sq_length:
            location = "cenR"
        else:
            location = "lowR"
    return location

def label_pose_and_location(img, x, y, w, h):
    threshold_img = convert_to_binary(img)
    pose_label = create_pose_label(img, x, y, w, h)
    location_label = create_location_label(img, x, y, w, h)

    # Combine the labels into a single string
    label = f"{pose_label}, {location_label}"

    # Display the combined label on the image
    cv.putText(img, label, (x - 20, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.imshow("Hand Detection", img)
    cv.imshow("Binary Intermediate", threshold_img)

def main():
    # # For step 1.2 1a)
    # path = './images/fist,cenC_0.jpg'

    # # For step 1.2 1b)
    # path = './images/fist,cenC_1.jpg'

    # # For step 1.2 2a)
    # path = './images/splay,uppR_0.jpg'

    # # For step 1.2 2b)
    # path = './images/splay,uppR_1.jpg'

    # # For step 1.3.1 1)
    # path = './images/fist,cenC_2.jpg'

    # # For step 1.3.1 2)
    # path = './images/unknown_as_fist_0.jpg'

    # # For step 1.3.1 3)
    # path = './images/splay,uppR_fn.jpg'

    # # For step 1.3.1 4)
    # path = './images/splay,uppR_fp.jpg'

    # # For step 1.3.1 5a)
    # path = './images/palm,cenC_0.jpg'

    # # For step 1.3.1 5b)
    # path = './images/palm,lowR_0.jpg'

    # # For step 1.3.2 1a)
    # path = './images/palm,uppL_0.jpg'

    # # For step 1.3.2 1b)
    # path = './images/palm,lowL_0.jpg'

    # # For step 1.3.2 2)
    # path = './images/palm_0.jpg'  # doesn't work
    # path = './images/palm_1.jpg'
    # path = './images/palm_2.jpg'
    # path = './images/palm_3.jpg'

    # # For step 1.3.2 3)
    # path = './images/splay_as_palm.jpg'

    # # For step 1.3.2 4a)
    # path = './images/unknown,uppR_0.jpg'

    # # For step 1.3.2 4b)
    # path = './images/unknown,uppR_1.jpg'

    # # For step 1.4 easy
    # path = './images/easy_me_0.jpg'
    # path = './images/easy_me_1.jpg'
    # path = './images/easy_friend_0.jpg'
    # path = './images/easy_friend_1.jpg'

    # # For step 1.4 good
    # path = './images/good_me_0.JPG'
    # path = './images/good_me_1.JPG'
    # path = './images/good_me_2.JPG'
    # path = './images/good_friend_0.JPG'
    # path = './images/good_friend_1.JPG'
    # path = './images/good_friend_2.JPG'

    # # For step 1.4 difficult
    # path = './images/diff_me_0.jpg'
    # path = './images/diff_me_1.jpg'
    # path = './images/diff_me_2.jpg'
    # path = './images/diff_me_3.jpg'
    # path = './images/diff_me_4.jpg'
    # path = './images/diff_friend_0.jpg'
    # path = './images/diff_friend_1.jpg'
    # path = './images/diff_friend_2.jpg'
    # path = './images/diff_friend_3.jpg'
    # path = './images/diff_friend_4.jpg'

    img = cv.imread(path)
    img = resize_img(img)
    x, y, w, h = detect_hand(img)
    label_pose_and_location(img, x, y, w, h)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
