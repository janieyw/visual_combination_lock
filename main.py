import cv2 as cv

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def detect_hand(image_path):
    image = cv.imread(image_path)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh_image = cv.threshold(gray_image, 160, 340, cv.THRESH_BINARY)
    contours_image, _ = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    hand_contour = max(contours_image, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(hand_contour)
    cv.rectangle(gray_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow(image_path, gray_image)
def main():
    fist_center_path = './images/fist,center.jpg'
    detect_hand(fist_center_path)
    splay_center_path = './images/splay,center.jpg'
    detect_hand(splay_center_path)
    palm_center_path = './images/palm,center.jpg'
    detect_hand(palm_center_path)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

# _, thresh0 = cv.threshold(gray, 180, 340, cv.THRESH_BINARY)
# cv.imshow('Threshold0', thresh0)
# _, thresh1 = cv.threshold(gray, 180, 350, cv.THRESH_BINARY)
# cv.imshow('Threshold1', thresh1)
# _, thresh3 = cv.threshold(gray, 170, 365, cv.THRESH_BINARY)
# cv.imshow('Threshold3', thresh3)
# _, thresh4 = cv.threshold(gray, 200, 400, cv.THRESH_BINARY)
# cv.imshow('Threshold4', thresh4)
# _, thresh5 = cv.threshold(gray, 100, 225, cv.THRESH_BINARY)
# cv.imshow('Threshold5', thresh5)
# _, thresh6 = cv.threshold(gray, 100, 225, cv.THRESH_BINARY)
# cv.imshow('Threshold6', thresh6)
# cv.imwrite('fist,center_threshold.jpg', thresh)
