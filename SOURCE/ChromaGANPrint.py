# Copyright (c) 2019 Patricia Vitoria <patricia.vitoria@upf.edu>,

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as
#published by the Free Software Foundation, either version 3 of the
#License, or (at your option) any later version.
#Smooth Contours is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU Affero General Public License for more details.
#You should have received a copy of the GNU Affero General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>. 

import os
import tensorflow as tf
import config as config
import numpy as np
import cv2
import dataClass as data
from keras import applications
from keras.models import load_model


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def reconstruct(batchX, predictedY, filelist):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    save_models_path = os.path.join(config.OUT_DIR,config.TEST_NAME)
    save_path = os.path.join(save_models_path, filelist +  "_reconstructed.jpg" )
    cv2.imwrite(save_path, result)
    return result

def reconstruct_no(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result

def sample_images():
    avg_cost = 0
    avg_cost2 = 0
    avg_cost3 = 0
    avg_ssim = 0
    avg_psnr = 0
    VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True) 
    save_models_path =os.path.join(config.MODEL_DIR,config.TEST_NAME)
    save_path = os.path.join(save_models_path, config.PRETRAINED)
    colorizationModel = load_model(save_path)
    test_data = data.DATA(config.TEST_DIR)
    total_batch = max(data.size,int(data.size/config.BATCH_SIZE))
    print("number of images to inpaint " + str(test_data.size))
    print("total number of batches to colorize " + str(total_batch))
    for b in range(total_batch):
            batchX, batchY,  filelist  = test_data.generate_batch()
            predY, _  = colorizationModel.predict(np.tile(batchX,[1,1,1,3]))
            predictVGG =VGG_modelF.predict(np.tile(batchX,[1,1,1,3]))
            loss = colorizationModel.evaluate(np.tile(batchX,[1,1,1,3]), [batchY, predictVGG], verbose=0)
            avg_cost += loss[0]
            avg_cost2 += loss[1]
            avg_cost3 += loss[2]
            for i in range(config.BATCH_SIZE):
                predResult = reconstruct(deprocess(batchX)[i], deprocess(predY)[i], filelist[i][:-4] )
                originalResult = reconstruct_no(deprocess(batchX)[i], deprocess(batchY)[i])
                avg_ssim += tf.keras.backend.eval( tf.image.ssim(tf.convert_to_tensor(originalResult, dtype=tf.float32), tf.convert_to_tensor(predResult, dtype=tf.float32), max_val=255))
                avg_psnr += tf.keras.backend.eval( tf.image.psnr(tf.convert_to_tensor(originalResult, dtype=tf.float32), tf.convert_to_tensor(predResult, dtype=tf.float32), max_val=255))
            print("Batch " + str(b)+"/"+str(total_batch))

    print(" ----------  loss =", "{:.8f}------------------".format(avg_cost/total_batch))
    print(" ----------  upsamplingloss =", "{:.8f}------------------".format(avg_cost2/total_batch))
    print(" ----------  classification_loss =", "{:.8f}------------------".format(avg_cost3/total_batch))
    print(" ----------  ssim loss =", "{:.8f}------------------".format(avg_ssim/(total_batch*config.BATCH_SIZE)))
    print(" ----------  psnr loss =", "{:.8f}------------------".format(avg_psnr/(total_batch*config.BATCH_SIZE)))



if __name__ == '__main__':
    sample_images()
