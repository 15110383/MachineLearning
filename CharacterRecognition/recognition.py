import preprocessing, train
import tensorflow as tf
import numpy as np
import cv2
import os


def recognition(filename):
    large = cv2.imread(filename)
    width, height, channel = large.shape
    cv2.imshow("Original", large)
    cv2.imwrite(os.path.join("img/", filename), large)
    rgb = cv2.pyrDown(large)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    (_, contours, hierarchy) = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    mask = cv2.resize(mask, (height, width))
    m = "mask" + filename
    cv2.imwrite(os.path.join("img/", m), mask)
    cv2.imshow("Mask", mask)
    cv2.waitKey(-1)
    predict(filename, "saves/saves/awesome_model-99000")
    cv2.waitKey(-1)


def open_image(filename, scale_to=[64, 64]):
    """Opens an image, returns the preprocessed image (scaled, masked)"""
    m = "img/" + filename
    print(m)
    n = "img/mask" + filename
    print(n)
    img = cv2.imread(m) * cv2.imread(n) / 255
    cv2.imshow("IMG", img)

    # scaling
    img = cv2.resize(img, tuple(scale_to))

    # normalising
    processed_img = img.astype(np.float32)
    for c in range(3):
        processed_img[:, :, c] /= np.max(processed_img[:, :, c])

    # to grayscale
    processed_img = cv2.cvtColor(
        (processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    processed_img = np.expand_dims(processed_img, -1)

    return processed_img


def predict(img, model_name):
    img_h, img_w = 64, 64

    nn = train.Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES))
    dataset = img

    n_test = len(dataset)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_name)

        image = open_image(dataset, (img_h, img_w))

        # predict
        classes = sess.run(nn.classes, feed_dict={nn.input: [image]})
        predicted_label = np.argmax(classes[0])
        print("Predicted %s" % preprocessing.CLASSES[predicted_label])


recognition("cap.jpg")
