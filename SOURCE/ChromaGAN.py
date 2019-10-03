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
import datetime
from functools import partial

import keras
from keras import applications
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import load_model, model_from_json, Model



GRADIENT_PENALTY_WEIGHT = 10  



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

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

    
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((config.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class MODEL():

    def __init__(self):
        self.dataset_name = config.DATASET
        self.img_shape_1 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
        self.img_shape_3 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
        self.img_shape_2 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 2)


        optimizer = Adam(0.00002, 0.5)
        self.discriminator = self.discriminator()
        self.discriminator.compile(loss=wasserstein_loss,
            optimizer=optimizer)

        self.colorizationModel = self.colorization_model()
        self.colorizationModel.compile(loss=['mse', 'kld'],
            optimizer=optimizer)

        img_Lab = Input(shape= self.img_shape_3)
        img_L = Input(shape= self.img_shape_1)
        batchX = Input(shape= self.img_shape_3)
        img_ab_real = Input(shape= self.img_shape_2)

        self.colorizationModel.trainable = False
        predAB, substracted = self.colorizationModel(img_Lab)
        discPredAB = self.discriminator([predAB, img_L])
        discriminator_output_from_real_samples = self.discriminator([img_ab_real, img_L])


        averaged_samples = RandomWeightedAverage()([img_ab_real,
                                            predAB])
        averaged_samples_out = self.discriminator([averaged_samples, img_L])


        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)

        partial_gp_loss.__name__ = 'gradient_penalty'


        self.discriminator_model = Model(inputs=[img_L, img_ab_real, img_Lab],
                            outputs=[discriminator_output_from_real_samples,
                                     discPredAB,
                                     averaged_samples_out])

        self.discriminator_model.compile(optimizer=optimizer,
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])



        self.colorizationModel.trainable = True
        self.discriminator.trainable = False
        self.combined = Model(inputs=[img_Lab, img_L],
                              outputs=[ predAB, substracted, discPredAB])

        self.combined.compile(loss=['mse','kld', wasserstein_loss],
                            loss_weights=[1.0, 0.003, -0.1],
                            optimizer=optimizer) #1/300
        self.log_path= os.path.join(config.LOG_DIR,config.TEST_NAME)
        if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
        self.callback = TensorBoard(self.log_path)
        self.callback.set_model(self.combined)
        self.train_names = ['loss', 'mse_loss', 'kullback_loss', 'wasserstein_loss']
        self.disc_names = ['disc_loss', 'disc_valid', 'disc_fake','disc_gp']


        self.test_loss_array = []
        self.g_loss_array = []


    def discriminator(self):
        input_ab = Input(shape=self.img_shape_2, name='ab_input')
        input_l = Input(shape=self.img_shape_1, name='l_input')
        net = keras.layers.concatenate([input_l, input_ab])
        net =  keras.layers.Conv2D(64, (4, 4), padding='same', strides=(2, 2))(net) # 112, 112, 64
        net = LeakyReLU()(net)
        net =  keras.layers.Conv2D(128, (4, 4), padding='same', strides=(2, 2))(net) # 56, 56, 128
        net = LeakyReLU()(net)
        net =  keras.layers.Conv2D(256, (4, 4), padding='same', strides=(2, 2))(net) # 28, 28, 256
        net = LeakyReLU()(net)
        net =  keras.layers.Conv2D(512, (4, 4), padding='same', strides=(1, 1))(net) # 28, 28, 512
        net = LeakyReLU()(net)
        net =  keras.layers.Conv2D(1, (4, 4), padding='same', strides=(1, 1))(net)  # 28, 28,1
        return Model([input_ab, input_l], net)




    def colorization_model(self):
        input_img = Input(shape=self.img_shape_3)

        # VGG16 Full
        VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True) # none, 1000
        for layer in VGG_modelF.layers:
            layer.trainable = False
        model2 = Model(VGG_modelF.input,VGG_modelF.output)
        VGG_modelFull = model2(input_img)

        # VGG16 without top
        VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        #for layer in VGG_model.layers:
        #    layer.trainable = False
        model2 = Model(VGG_model.input,VGG_model.layers[-6].output)
        model = model2(input_img)


        # Global Features

        global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(model)
        global_features = keras.layers.BatchNormalization()(global_features)
        global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)

        global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)
        global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)


        global_features2 = keras.layers.Flatten()(global_features)
        global_features2 = keras.layers.Dense(1024)(global_features2)
        global_features2 = keras.layers.Dense(512)(global_features2)
        global_features2 = keras.layers.Dense(256)(global_features2)
        global_features2 = keras.layers.RepeatVector(28*28)(global_features2)
        global_features2 = keras.layers.Reshape((28,28, 256))(global_features2)



        global_featuresClass = keras.layers.Flatten()(global_features)
        global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
        global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
        global_featuresClass = keras.layers.Dense(1000, activation='softmax')(global_featuresClass)



        midlevel_features = keras.layers.Conv2D(512, (3, 3),  padding='same', strides=(1, 1), activation='relu')(model)
        midlevel_features = keras.layers.BatchNormalization()(midlevel_features)
        midlevel_features = keras.layers.Conv2D(256, (3, 3),  padding='same', strides=(1, 1), activation='relu')(midlevel_features)
        midlevel_features = keras.layers.BatchNormalization()(midlevel_features)

        # fusion of (VGG16 + Midlevel) + (VGG16 + Global)
        modelFusion = keras.layers.concatenate([midlevel_features, global_features2])

        # Fusion + Colorization
        outputModel =  keras.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(modelFusion)
        outputModel =  keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)

        outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        outputModel =  keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        outputModel =  keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)

        outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        outputModel =  keras.layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        outputModel =  keras.layers.Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')(outputModel)
        outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        final_model = Model(input=input_img, outputs = [outputModel, global_featuresClass])

        return final_model


    def train(self, data, log,sample_interval=1):
        VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True) # none, 1000
        positive_y = np.ones((config.BATCH_SIZE, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((config.BATCH_SIZE, 1), dtype=np.float32)
        total_batch = int(data.size/config.BATCH_SIZE)
        it = 0
        save_models_path =os.path.join(config.MODEL_DIR,config.TEST_NAME)
        if not os.path.exists(save_models_path):
                os.makedirs(save_models_path)
        for epoch in range(config.NUM_EPOCHS):
                for batch in range(int(data.size/config.BATCH_SIZE)):
                    batchX, batchY, _ = data.generate_batch()
                    l_3=np.tile(batchX,[1,1,1,3])
                    predictVGG =VGG_modelF.predict(l_3)
                    g_loss_col =self.combined.train_on_batch([l_3, batchX],
                                                        [batchY, predictVGG, positive_y])
                    write_log(self.callback, self.train_names, g_loss_col, it)
                    it = it+1
                    d_loss_real = self.discriminator_model.train_on_batch([batchX, batchY, l_3], [positive_y, negative_y, dummy_y])
                    write_log(self.callback, self.disc_names, d_loss_real, it)
                    if (batch+1)%1000 ==0: 
                        print("[Epoch %d] [Batch %d/%d] [loss: %08f]" %  ( epoch, batch,total_batch, g_loss_col[0]))
                save_path = os.path.join(save_models_path, "my_model_combinedEpoch%d.h5" % epoch)
                self.combined.save(save_path) 
                save_path = os.path.join(save_models_path, "my_model_colorizationEpoch%d.h5" % epoch)
                self.colorizationModel.save(save_path) 
                save_path = os.path.join(save_models_path, "my_model_discriminatorEpoch%d.h5" % epoch)
                self.discriminator.save(save_path)  
                self.sample_images()


    def sample_images(self):
        for _ in range(total_batch):
                batchX, batchY,  filelist  = test_data.generate_batch()
                predY, _  = self.colorizationModel.predict(np.tile(batchX,[1,1,1,3]))
                predictVGG =VGG_modelF.predict(np.tile(batchX,[1,1,1,3]))
                for i in range(config.BATCH_SIZE):
                        predResult = reconstruct(deprocess(batchX)[i], deprocess(predY)[i], filelist[i][:-4] )



if __name__ == '__main__':
    log_path= os.path.join(config.LOG_DIR,config.TEST_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".txt"), "w") as log:
        log.write(str(datetime.datetime.now()) + "\n")
        log.write("Use Pretrained Weights: " + str(config.USE_PRETRAINED) + "\n")
        print("Use Pretrained Weights"  + str(config.USE_PRETRAINED))
        log.write("Pretrained Model: " + config.PRETRAINED + "\n")
        print("Pretrained Model")
        train_data = data.DATA(config.TRAIN_DIR)
        print("Train data loaded")
        colorizationModel = MODEL()
        print("Model Initialized")
        print("Start training")
        colorizationModel.train(train_data, log)
