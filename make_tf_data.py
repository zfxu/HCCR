import cv2
import numpy as np
import random
import os
import tensorflow as tf

CHARSET_SIZE = 3755
IMG_SIZE = 64
BASE_SIZE = 56


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_data(data_dir):
    # 获取所有图像的名字和标签
    truncate_path = data_dir + ('%05d' % CHARSET_SIZE)
    print(truncate_path)
    image_names = []
    for root, sub_folder, file_list in os.walk(data_dir):
        if root < truncate_path:
            image_names += [os.path.join(root, file_path) for file_path in file_list]
    random.shuffle(image_names)
    labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in image_names]

    print(len(labels))

    count = 0
    num_file = 50000
    while True:
        i = int(count / num_file)
        # file_name = '/opt/' + data_dir[2:-1] + str(i) + '.tfrecords'
        file_name = data_dir[2:-1] + str(i) + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(file_name)

        for j in range(num_file):
            n = i * num_file + j
            if count == len(labels):
                break
            img = cv2.imread(image_names[n], 0)
            if img is not None:
                img = 255 - img
                img = pre_process(img)

                image_raw = img.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(int(labels[n])),
                    'image_raw': _bytes_feature(image_raw)}))
                writer.write(example.SerializeToString())
                count += 1
            if count % 5000 == 0:
                print(count)
        if count == len(labels):
            break


def pre_process(img):
    # 这里背景要是黑色的
    rect = cv2.boundingRect(img)
    width = rect[2]
    height = rect[3]
    img = img[rect[1]:rect[1]+height, rect[0]:rect[0]+width]

    res = np.zeros((IMG_SIZE, IMG_SIZE), dtype='uint8')*255

    if width > height:
        height = BASE_SIZE * height // width
        width = BASE_SIZE
    else:
        width = BASE_SIZE * width // height
        height = BASE_SIZE

    # 选取特定区域
    img = cv2.resize(img, (width, height))
    height_start = (IMG_SIZE - height) // 2
    width_start = (IMG_SIZE - width) // 2
    res[height_start:height_start+height, width_start:width_start+width] = img

    # 归一化到0~255
    res = 255/(res.max()-res.min())*(res-res.min())

    # 灰度反转 二值化
    _, res = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)
    res = res.astype('uint8')

    # cv2.imshow("", res)
    # cv2.waitKey(0)
    return res


def gabor_filter(img):
    img_out = np.zeros((img.shape[0], img.shape[1], 9), dtype='uint8')
    img_out[:,:,0] = img

    ksize = 5
    theta_list = np.arange(0, 180, 22.5) / 360 * 2 * np.pi
    n = 1
    for i in theta_list:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma=3.14, theta=i, lambd=0.5, gamma=0.5, psi=0)
        res = cv2.filter2D(img, -1, kernel)
        img_out[:,:,n] = res
        n += 1

    return img_out


if __name__ == '__main__':
    make_data('./train/')
    make_data('./test/')
    # img = cv2.imread("1.png", 0)
    # pre_process(img)