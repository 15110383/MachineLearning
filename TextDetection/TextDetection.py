import cv2
import numpy as np

def DetectText(filename):
    #Create MSER object
    mser = cv2.MSER_create()

    #Your image path i-e receipt path
    img = cv2.imread(filename)

    #Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vis = img.copy()

    #detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    cv2.imshow('img', vis)

    cv2.waitKey(0)

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for contour in hulls:

        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

        # [x, y, w, h] = cv2.boundingRect(contour)
        # if w < 35 and h < 35:
        #     continue
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)

    #this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("text-only", text_only)

    cv2.waitKey(0)



DetectText('Image.png')


# import numpy as np
# import cv2
#
# ## Read image and change the color space
# imgname = "C:/Users/Thien Doan/Desktop/Smile_Horizonal_2_1024x.png"
# img = cv2.imread(imgname)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ## Get mser, and set parameters
# mser = cv2.MSER_create()
# mser.setMinArea(100)
# mser.setMaxArea(800)
#
# ## Do mser detection, get the coodinates and bboxes
# coordinates, bboxes = mser.detectRegions(gray)
#
# ## Filter the coordinates
# vis = img.copy()
# coords = []
# for coord in coordinates:
#     bbox = cv2.boundingRect(coord)
#     x,y,w,h = bbox
#     if w< 10 or h < 10 or w/h > 5 or h/w > 5:
#         continue
#     coords.append(coord)
#
# ## colors
# colors = [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200], [43, 169, 200], [43, 200, 195], [43, 200, 163], [43, 200, 132], [43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43], [116, 200, 43], [148, 200, 43], [179, 200, 43], [200, 184, 43], [200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43], [200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158], [200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200], [80, 43, 200], [43, 43, 200]]
#
# ## Fill with random colors
# np.random.seed(0)
# canvas1 = img.copy()
# canvas2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# canvas3 = np.zeros_like(img)
#
# for cnt in coords:
#     xx = cnt[:,0]
#     yy = cnt[:,1]
#     color = colors[np.random.choice(len(colors))]
#     canvas1[yy, xx] = color
#     canvas2[yy, xx] = color
#     canvas3[yy, xx] = color
#
# ## Save
# cv2.imwrite("result1.png", canvas1)
# cv2.imwrite("result2.png", canvas2)
# cv2.imwrite("result3.png", canvas3)
