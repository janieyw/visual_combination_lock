import cv2 as cv
import numpy as np
def resize_img(img):  # resize to 450px X 450px
    height, width = img.shape[:2]
    scaling_factor = 450.0 / height
    dimensions = (int(width * scaling_factor), 450)
    return cv.resize(img, dimensions, interpolation = cv.INTER_AREA)

def convert_to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def convert_to_binary(img):
    gray_img = convert_to_gray(img)
    _, thresh_img = cv.threshold(gray_img, 160, 340, cv.THRESH_BINARY)
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
    contours_img, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
    hand_contour = max(contours_img, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(hand_contour)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow("Hand Detection", img)
    return x, y, w, h

# def create_pose_label(img, x, y, w, h):
#     return hand_pose

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
            location = "upper_left"
        elif center_y < 2 * sq_length:
            location = "none"  # middle_left
        else:
            location = "lower_left"
    elif center_x < 2 * sq_length:
        if center_y < sq_length:
            location = "none"  # lower_center
        elif center_y < 2 * sq_length:
            location = "center"  # middle_center
        else:
            location = "none"  # upper_center
    else:
        if center_y < sq_length:
            location = "upper_right"
        elif center_y < 2 * sq_length:
            location = "none"  # middle_right
        else:
            location = "lower_right"

    cv.putText(img, location, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.imshow("Hand Detection", img)
    # return location

# def label_pose_and_location(img, x, y, w, h):
#     pose_label = create_pose_label(img, x, y, w, h)
#     location_label = create_location_label(img, x, y, w, h)
#
#     # Combine the labels into a single string
#     label = f"{pose_label}, {location_label}"
#
#     # Display the combined label on the image
#     cv.putText(img, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
#     cv.imshow("Hand Detection", img)

def main():
    # path_fist_center = './images/fist,center.jpg'
    # img_fist_center = cv.imread(path_fist_center)
    # img_fist_center = resize_img(img_fist_center)
    # # img_fist_center = convert_to_binary(img_fist_center)
    # x, y, w, h = detect_hand(img_fist_center)
    # create_location_label(img_fist_center, x, y, w, h)

    # path_splay_uppR = './images/splay,uppR.jpg'
    # img_splay_uppR = cv.imread(path_splay_uppR)
    # img_splay_uppR = resize_img(img_splay_uppR)
    # img_splay_uppR = darken_non_red_regions(img_splay_uppR)
    # x, y, w, h = detect_hand(img_splay_uppR)
    # create_location_label(img_splay_uppR, x, y, w, h)

    # path_fist_uppL = './images/fist,uppL.jpg'
    # img_fist_uppL = cv.imread(path_fist_uppL)
    # img_fist_uppL = resize_img(img_fist_uppL)
    # x, y, w, h = detect_hand(img_fist_uppL)
    # create_location_label(img_fist_uppL, x, y, w, h)

    # path_splay_lowL = './images/splay,lowL.jpg'
    # img_splay_lowL = cv.imread(path_splay_lowL)
    # img_splay_lowL = resize_img(img_splay_lowL)
    # x, y, w, h = detect_hand(img_splay_lowL)
    # create_location_label(img_splay_lowL, x, y, w, h)

    # path_palm_lowR = './images/palm,lowR.jpg'
    # img_palm_lowR = cv.imread(path_palm_lowR)
    # img_palm_lowR = resize_img(img_palm_lowR)
    # x, y, w, h = detect_hand(img_palm_lowR)
    # create_location_label(img_palm_lowR, x, y, w, h)

    # path_fist_lowL = './images/fist,lowL.jpg'
    # img_fist_lowL = cv.imread(path_fist_lowL)
    # img_fist_lowL = resize_img(img_fist_lowL)
    # x, y, w, h = detect_hand(img_fist_lowL)
    # create_location_label(img_fist_lowL, x, y, w, h)

    path_palm_uppL = './images/palm,uppL.jpg'
    img_palm_uppL = cv.imread(path_palm_uppL)
    img_palm_uppL = resize_img(img_palm_uppL)
    x, y, w, h = detect_hand(img_palm_uppL)
    create_location_label(img_palm_uppL, x, y, w, h)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
