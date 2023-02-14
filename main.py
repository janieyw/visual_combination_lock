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
    thresh_img = cv.erode(thresh_img, None, iterations=2)
    thresh_img = cv.dilate(thresh_img, None, iterations=2)
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
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(max_contour)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow("Hand Detection", img)
    return x, y, w, h

# referenced Izane's Github (https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python/blob/master/new.py)
def count_fingers(masked_frame, drawing):
    hull = cv.convexHull(masked_frame, returnPoints=False)
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

    # Determine which of the 9 smaller squares the center of the bounding
    # rectangle falls into
    if center_x < sq_length:
        if center_y < sq_length:
            location = "upperL"
        elif center_y < 2 * sq_length:
            location = "centerL"
        else:
            location = "lowerL"
    elif center_x < 2 * sq_length:
        if center_y < sq_length:
            location = "lowerC"
        elif center_y < 2 * sq_length:
            location = "centerC"
        else:
            location = "upperC"
    else:
        if center_y < sq_length:
            location = "upperR"
        elif center_y < 2 * sq_length:
            location = "centerR"
        else:
            location = "lowerR"
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
    # # For step 1.2 1)
    # path_fist_center = './images/fist,center.jpg'
    # img_fist_center = cv.imread(path_fist_center)
    # img_fist_center = resize_img(img_fist_center)
    # x, y, w, h = detect_hand(img_fist_center)
    # label_pose_and_location(img_fist_center, x, y, w, h)

    # # For step 1.2 2)
    # path_fist_centerC = './images/fist,centerC.jpg'
    # img_fist_centerC = cv.imread(path_fist_centerC)
    # img_fist_centerC = resize_img(img_fist_centerC)
    # x, y, w, h = detect_hand(img_fist_centerC)
    # label_pose_and_location(img_fist_centerC, x, y, w, h)

    # # For step 1.2 3)
    # path_splay_uppR = './images/splay,uppR.jpg'
    # img_splay_uppR = cv.imread(path_splay_uppR)
    # img_splay_uppR = resize_img(img_splay_uppR)
    # x, y, w, h = detect_hand(img_splay_uppR)
    # label_pose_and_location(img_splay_uppR, x, y, w, h)

    # # For step 1.2 4)
    # path_splay_uppR = './images/splay,uppR_0.jpg'
    # img_splay_uppR = cv.imread(path_splay_uppR)
    # img_splay_uppR = resize_img(img_splay_uppR)
    # x, y, w, h = detect_hand(img_splay_uppR)
    # label_pose_and_location(img_splay_uppR, x, y, w, h)

    # # For step 1.3.1 1)
    # path_fist_centerC = './images/fist,centerC_1.jpg'
    # img_fist_centerC = cv.imread(path_fist_centerC)
    # img_fist_centerC = resize_img(img_fist_centerC)
    # x, y, w, h = detect_hand(img_fist_centerC)
    # label_pose_and_location(img_fist_centerC, x, y, w, h)

    # # For step 1.3.1 2)
    # path_fist_centerC = './images/fist,centerC_2.jpg'
    # img_fist_centerC = cv.imread(path_fist_centerC)
    # img_fist_centerC = resize_img(img_fist_centerC)
    # x, y, w, h = detect_hand(img_fist_centerC)
    # label_pose_and_location(img_fist_centerC, x, y, w, h)

    # # For step 1.3.1 3)
    # path_splay_uppR_fn = './images/splay,uppR_fn.jpg'
    # img_splay_uppR_fn = cv.imread(path_splay_uppR_fn)
    # img_splay_uppR_fn = resize_img(img_splay_uppR_fn)
    # x, y, w, h = detect_hand(img_splay_uppR_fn)
    # label_pose_and_location(img_splay_uppR_fn, x, y, w, h)

    # # For step 1.3.1 4)
    # path_splay_uppR_fp = './images/splay,uppR_fp.jpg'
    # img_splay_uppR_fp = cv.imread(path_splay_uppR_fp)
    # img_splay_uppR_fp = resize_img(img_splay_uppR_fp)
    # x, y, w, h = detect_hand(img_splay_uppR_fp)
    # label_pose_and_location(img_splay_uppR_fp, x, y, w, h)

    # # # For step 1.3.1 5a)
    # path_palm_centerC = './images/palm,center.jpg'
    # img_palm_centerC = cv.imread(path_palm_centerC)
    # img_palm_centerC = resize_img(img_palm_centerC)
    # x, y, w, h = detect_hand(img_palm_centerC)
    # label_pose_and_location(img_palm_centerC, x, y, w, h)

    # # For step 1.3.1 5b)
    # path_palm_lowR = './images/palm,lowR.jpg'
    # img_palm_lowR = cv.imread(path_palm_lowR)
    # img_palm_lowR = resize_img(img_palm_lowR)
    # x, y, w, h = detect_hand(img_palm_lowR)
    # label_pose_and_location(img_palm_lowR, x, y, w, h)

    # # For step 1.3.2 1a)
    # path_palm_uppL = './images/palm,uppL.jpg'
    # img_palm_uppL = cv.imread(path_palm_uppL)
    # img_palm_uppL = resize_img(img_palm_uppL)
    # x, y, w, h = detect_hand(img_palm_uppL)
    # label_pose_and_location(img_palm_uppL, x, y, w, h)

    # # For step 1.3.2 1b)
    # path_palm_lowL = './images/palm,lowL.jpg'
    # img_palm_lowL = cv.imread(path_palm_lowL)
    # img_palm_lowL = resize_img(img_palm_lowL)
    # x, y, w, h = detect_hand(img_palm_lowL)
    # label_pose_and_location(img_palm_lowL, x, y, w, h)

    # # For step 1.3.2 2)
    # path_palm_cenC = './images/palm_1.jpg'
    # img_palm_cenC = cv.imread(path_palm_cenC)
    # img_palm_cenC = resize_img(img_palm_cenC)
    # x, y, w, h = detect_hand(img_palm_cenC)
    # label_pose_and_location(img_palm_cenC, x, y, w, h)

    # # For step 1.3.2 3)
    # path_splay_as_palm = './images/splay_as_palm.jpg'
    # img_splay_as_palm = cv.imread(path_splay_as_palm)
    # img_splay_as_palm = resize_img(img_splay_as_palm)
    # x, y, w, h = detect_hand(img_splay_as_palm)
    # label_pose_and_location(img_splay_as_palm, x, y, w, h)

    # # For step 1.3.2 4a)
    # path_other_uppR = './images/other,uppR.jpg'
    # img_other_uppR = cv.imread(path_other_uppR)
    # img_other_uppR = resize_img(img_other_uppR)
    # x, y, w, h = detect_hand(img_other_uppR)
    # label_pose_and_location(img_other_uppR, x, y, w, h)

    # # For step 1.3.2 4b)
    # path_other_uppR = './images/other,uppR_0.jpg'
    # img_other_uppR = cv.imread(path_other_uppR)
    # img_other_uppR = resize_img(img_other_uppR)
    # x, y, w, h = detect_hand(img_other_uppR)
    # label_pose_and_location(img_other_uppR, x, y, w, h)

    # # For step 1.4 easy
    # path_easy = './images/easyf_2.jpg'
    # img_easy = cv.imread(path_easy)
    # img_easy = resize_img(img_easy)
    # x, y, w, h = detect_hand(img_easy)
    # label_pose_and_location(img_easy, x, y, w, h)

    # # For step 1.4 good
    # path_easy = './images/goodf_3.JPG'
    # img_easy = cv.imread(path_easy)
    # img_easy = resize_img(img_easy)
    # x, y, w, h = detect_hand(img_easy)
    # label_pose_and_location(img_easy, x, y, w, h)

    # # For step 1.4 difficult
    # path_diff = './images/difff_5.jpg'
    # img_diff = cv.imread(path_diff)
    # img_diff = resize_img(img_diff)
    # x, y, w, h = detect_hand(img_diff)
    # label_pose_and_location(img_diff, x, y, w, h)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
