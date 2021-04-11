# ChromaGAN
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2
import math
import argparse

parser = argparse.ArgumentParser(description="Chrome GAN")
parser.add_argument("--input_dir", type=str, default="./input_images", help="directory with images to be transformed")
parser.add_argument("--output_dir", type=str, default="./output_images", help="directory where transformed images are saved")
parser.add_argument("--model", type=str, default="./MODEL/my_model_colorization.h5", help="directory with the pretrained model")
parser.add_argument("--batch_size", type=int, default=1, help="number of images that will be transformed in parallel to speed up processing.")
args = parser.parse_args()


def read_img(filename):
    image_size = 224
    max_side = 1500
    img = cv2.imread(filename, 3)
    if img is None:
        print("Unable to read image: " + filename)
        return False, False, False, False, False
    height, width, channels = img.shape
    if height > max_side or width > max_side:
        print("Image " + filename + " is of size (" + str(height) + "," + str(width) + ").")
        print("The maximum image size allowed is (" + str(max_side) + "," + str(max_side) + ").")
        r = min(max_side / height, max_side / width)
        height = math.floor(r * height)
        width = math.floor(r * width)
        img = cv2.resize(img, (width, height))
        print("It has been resized to (" + str(height) + "," + str(width) + ")")
    labimg = cv2.cvtColor(cv2.resize(img, (image_size, image_size)), cv2.COLOR_BGR2Lab)
    labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    return True, np.reshape(labimg[:, :, 0], (image_size, image_size, 1)), labimg[:, :, 1:], img, np.reshape(labimg_ori[:, :, 0], (height, width, 1))


class DATA:
    def __init__(self, dirname):
        self.dir_path = dirname
        self.file_list = os.listdir(self.dir_path)
        self.batch_size = args.batch_size
        self.size = len(self.file_list)
        self.data_index = 0

    def generate_batch(self):
        batch = []
        labels = []
        file_list = []
        labimg_oritList = []
        original_list = []
        for i in range(self.batch_size):
            filename = os.path.join(self.dir_path, self.file_list[self.data_index])
            ok, greyimg, colorimg, original, labimg_ori = read_img(filename)
            if ok:
                file_list.append(self.file_list[self.data_index])
                batch.append(greyimg)
                labels.append(colorimg)
                original_list.append(original)
                labimg_oritList.append(labimg_ori)
                self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch) / 255  # values between 0 and 1
        labels = np.asarray(labels) / 255  # values between 0 and 1
        original_list = np.asarray(original_list)
        labimg_oritList = np.asarray(labimg_oritList) / 255
        return batch, labels, file_list, original_list, labimg_oritList


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)

    return result


def reconstruct_each_img(batch_size, batch_info, pred_y):
    batch_x, batch_y, file_list, original, labimg_orit_list = batch_info
    for i in range(batch_size):
        original_result = original[i]
        height, width, channels = original_result.shape
        pred_y_2 = deprocess(pred_y[i])
        pred_y_2 = cv2.resize(pred_y_2, (width, height))
        labimg_orit_list_2 = labimg_orit_list[i]
        pred_result_2 = reconstruct(deprocess(labimg_orit_list_2), pred_y_2)
        save_path = os.path.join(args.output_dir, file_list[i][:-4] + "_reconstructed.jpg")
        cv2.imwrite(save_path, pred_result_2)


def colorize():
    save_path = os.path.join(args.model)
    colorization_model = load_model(save_path)
    test_data = DATA(args.input_dir)
    assert test_data.size >= 0, "Your list of images to colorize is empty. Please load images."
    assert args.batch_size <= test_data.size, "The batch size (" + str(args.batch_size) + ") should be smaller or equal to the number of images (" + str(test_data.size) + ")"
    total_batch = int(test_data.size / args.batch_size)
    print("")
    print("number of images to colorize: " + str(test_data.size))
    print("total number of batches to colorize: " + str(total_batch))
    print("")
    if not os.path.exists(args.output_dir):
        print('created save result path')
        os.makedirs(args.output_dir)
    for _ in tqdm(range(total_batch)):
        batch_info = test_data.generate_batch()
        batch_x = batch_info[0]
        if batch_x.any():
            pred_y, _ = colorization_model.predict(np.tile(batch_x, [1, 1, 1, 3]), use_multiprocessing=True)
            reconstruct_each_img(args.batch_size, batch_info, pred_y)


colorize()
