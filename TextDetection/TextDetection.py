import cv2
import numpy as np


def DetectAndSegment(filename):
    img = cv2.imread(filename, -1)
    img2 = img.copy()
    cv2.imshow("original.jpg", img)
    cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # --- Black image to be used to draw individual convex hull ---
    black = np.zeros_like(img)
    # cv2.imshow("black.jpg", black)

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[
        0])  # added by OP : this sorts contours left to right, so images come in order

    for cnt in contours:
        hull = cv2.convexHull(cnt)

        img3 = img.copy()
        black2 = black.copy()

        # --- Here is where I am filling the contour after finding the convex hull ---
        cv2.drawContours(black2, [hull], -1, (255, 255, 255), -1)
        g2 = cv2.cvtColor(black2, cv2.COLOR_BGR2GRAY)
        r, t2 = cv2.threshold(g2, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("t2.jpg", t2)

        masked = cv2.bitwise_and(img2, img2, mask=t2)
        cv2.imshow("masked.jpg", masked)

        print(len(hull))
        cv2.waitKey(0)

    cv2.destroyAllWindows()


DetectAndSegment('C:\\Users\\Thien Doan\\Desktop\\TextDetection\\Image.jpg')