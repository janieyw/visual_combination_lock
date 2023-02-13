import cv2 as cv

def detect_hand(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh_image = cv.threshold(gray_image, 160, 340, cv.THRESH_BINARY)
    contours_image, _ = cv.findContours(thresh_image, cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
    hand_contour = max(contours_image, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(hand_contour)
    return x, y, w, h

def label_location(image, x, y, w, h):
    # Get the length of the image
    image_length = image.shape[0]

    # Calculate the length of each 3x3 square within the image
    sq_length = image_length // 3

    # Identify the center of the bounding rectangle surrounding the hand
    center_x, center_y = x + w // 2, y + h // 2

    # Determine which of the 9 smaller squares the center of the bounding
    # rectangle falls into
    if center_x < sq_length:
        if center_y < sq_length:
            location = "lower_left"
        elif center_y < 2 * sq_length:
            location = "none"  # middle_left
        else:
            location = "upper_left"
    elif center_x < 2 * sq_length:
        if center_y < sq_length:
            location = "none"  # lower_center
        elif center_y < 2 * sq_length:
            location = "center"  # middle_center
        else:
            location = "none"  # upper_center
    else:
        if center_y < sq_length:
            location = "lower_right"
        elif center_y < 2 * sq_length:
            location = "none"  # middle_right
        else:
            location = "upper_right"

    # Display the label on the image above the rectangle
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(image, location, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Hand Detection", image)

def main():
    path_fist_center_ = './images/splay,center.jpg'
    image_fist_center = cv.imread(path_fist_center_)
    x, y, w, h = detect_hand(image_fist_center)
    label_location(image_fist_center, x, y, w, h)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
