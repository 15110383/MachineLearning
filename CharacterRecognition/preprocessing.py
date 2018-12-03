import cv2
import argparse
import os
import string, re
import numpy as np
import random

CLASSES = string.digits + string.ascii_uppercase + string.ascii_lowercase


def load_filenames(datapath, filters=[]):
    filenames = []
    for path, dirs, files in os.walk(datapath):
        # Lọc các path không đúng
        # filter = ['Good', 'Bmp'], nếu filename có trong 2 folder 'Good' 'Bmp' thì sẽ lấy.
        if sum(map(lambda f: f in path, filters)) == len(filters):
            filenames += list(map(lambda f: path + '/' + f, files))
    return filenames


def split_and_save_dataset(dataset, filename):
    # Tach dataset thanh 3 phan. Save list filename
    splits = [0.7, 0.1, 0.2]
    split_names = ['train', 'validation', 'test']
    perm = np.random.permutation(len(dataset))

    for s, split in enumerate(splits):
        startindex = int(sum(splits[:s]) * len(dataset))
        endindex = int(startindex + splits[s] * len(dataset))
        with open(filename + '_' + split_names[s], 'w') as f:
            for i in perm[startindex:endindex]:
                f.write(dataset[i] + '\n')

# Lấy index của ký tự
def get_class_index(filename):
    return int(re.findall(r'.*img(\d+).*', filename)[0]) - 1

# Lấy ký tự được dự đoán từ index
def get_class(filename):
    return CLASSES[get_class_index(filename)]


def get_batch(dataset, batch_size, dimensions):
    # Lấy số lượng (batch_size) từ dataset.
    batch_filenames = random.sample(dataset, batch_size)
    # List image
    images = np.array(list(map(lambda f: open_image(f, dimensions), batch_filenames)))
    # List label
    labels = np.array(list(map(get_class_index, batch_filenames)))
    return images, labels


def open_image(filename, scale_to=[64, 64]):
    # Đọc ảnh gốc và mask của 1 ký tự
    img = cv2.imread(filename) * cv2.imread(filename.replace('Bmp', 'Msk')) / 255

    # Resize thành 64x64
    img = cv2.resize(img, tuple(scale_to))

    # chuẩn hóa
    processed_img = img.astype(np.float32)
    for c in range(3):
        processed_img[:, :, c] /= np.max(processed_img[:, :, c])

    # grayscale
    processed_img = cv2.cvtColor(
        (processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    processed_img = np.expand_dims(processed_img, -1)

    return processed_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print(parser)
    parser.add_argument('datapath')
    parser.add_argument('-s', default="", help='Split and save dataset')
    parser.add_argument('-t', action='store_true', help='Print dataset stats')

    opt = parser.parse_args()

    if opt.s:
        filenames = load_filenames(opt.datapath, ['Good', 'Bmp'])
        split_and_save_dataset(filenames, opt.s)