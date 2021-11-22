# Copyright (c) 2019 Patricia Vitoria <patricia.vitoria@upf.edu>,

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# Smooth Contours is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import tensorflow as tf
import numpy as np
import cv2
import datetime
from functools import partial

import config as config
import dataClass as data
import transformerBlocks as trans
import wrappedDiscModel as wrapper

import keras
from keras import applications
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import load_model, model_from_json, Model

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):

    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    save_results_path = os.path.join(config.OUT_DIR, config.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(
        save_results_path, filelist + "_reconstructed.jpg")
    cv2.imwrite(save_path, result)
    return result


def reconstruct_no(batchX, predictedY):

    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result


def write_log(callback, names, logs, batch_no):
    pass
    # for name, value in zip(names, logs):
    #     summary = tf.Summary()
    #     summary_value = summary.value.add()
    #     summary_value.simple_value = value
    #     summary_value.tag = name
    #     callback.writer.add_summary(summary, batch_no)
    #     callback.writer.flush()


def wasserstein_loss(y_true, y_pred):

    return tf.reduce_mean(y_pred)


class MODEL():

    def __init__(self):

        self.img_shape_1 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
        self.img_shape_2 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 2)
        self.img_shape_3 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)

        optimizer = Adam(0.00002, 0.5)
        self.discriminator = self.discriminator()
        self.discriminator.compile(loss=wasserstein_loss,
                                   optimizer=optimizer)

        self.colorizationModel = self.colorization_model()
        self.colorizationModel.compile(loss=['mse', 'kld'],
                                       optimizer=optimizer)

        img_L_3 = Input(shape=self.img_shape_3)
        img_L = Input(shape=self.img_shape_1)
        img_ab_real = Input(shape=self.img_shape_2)

        self.colorizationModel.trainable = False
        predAB, classVector = self.colorizationModel(img_L_3)
        discPredAB = self.discriminator([predAB, img_L])
        discriminator_output_from_real_samples = self.discriminator(
            [img_ab_real, img_L])

        # averaged_samples = RandomWeightedAverage()([img_ab_real,
        #                                             predAB])
        # averaged_samples_out = self.discriminator([averaged_samples, img_L])
        # partial_gp_loss = partial(gradient_penalty_loss,
        #                           averaged_samples=averaged_samples,
        #                           gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        # partial_gp_loss.__name__ = 'gradient_penalty'

        # self.discriminator_model = Model(inputs=[img_L, img_ab_real, predAB],
        #                                  outputs=[discriminator_output_from_real_samples,
        #                                           discPredAB,
        #                                           averaged_samples_out])

        # self.discriminator_model.compile(optimizer=optimizer,
        #                                  loss=[wasserstein_loss,
        #                                        wasserstein_loss,
        #                                        partial_gp_loss], loss_weights=[-1.0, 1.0, 1.0])

        self.discriminator_model = wrapper.WrappedDiscriminatorModel(inputs=[img_L, img_ab_real, predAB],
                                                                     outputs=[discriminator_output_from_real_samples,
                                                                              discPredAB],
                                                                     discriminator=self.discriminator)

        self.discriminator_model.compile(optimizer=optimizer)

        self.colorizationModel.trainable = True
        self.discriminator.trainable = False
        self.combined = Model(inputs=[img_L_3, img_L],
                              outputs=[predAB, classVector, discPredAB])
        self.combined.compile(loss=['mse', 'kld', wasserstein_loss],
                              loss_weights=[1.0, 0.003, -0.1],
                              optimizer=optimizer)  # 1/300

        self.log_path = os.path.join(config.LOG_DIR, config.TEST_NAME)
        self.callback = TensorBoard(self.log_path)
        self.callback.set_model(self.combined)
        self.train_names = ['loss', 'mse_loss',
                            'kullback_loss', 'wasserstein_loss']
        self.disc_names = ['disc_loss', 'disc_valid', 'disc_fake', 'disc_gp']

        self.test_loss_array = []
        self.g_loss_array = []

    def discriminator_new(self):

        input_ab = Input(shape=self.img_shape_2, name='ab_input')
        input_l = Input(shape=self.img_shape_1, name='l_input')
        net = keras.layers.concatenate([input_l, input_ab])

        patch_size = 8
        num_patches = int(config.IMAGE_SIZE/patch_size) ** 2
        embedding_dimensions = 64
        num_heads = 8

        patches = trans.Patches(patch_size)(net)
        encoded_patches = trans.PatchEncoder(
            num_patches, embedding_dimensions)(patches)

        for _ in range(4):
            # Layer normalization 1.
            x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.

            # attention_output = keras.layers.MultiHeadAttention(
            #     num_heads=num_heads, key_dim=embedding_dimensions, dropout=0.1
            # )(x1, x1)
            attention_output = x1

            # Skip connection 1.
            x2 = keras.layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = keras.layers.Dense(
                2*embedding_dimensions, activation=tf.nn.gelu)(x3)
            # x3 = keras.layers.Dropout(0.5)(x3)
            x3 = keras.layers.Dense(
                embedding_dimensions, activation=tf.nn.gelu)(x3)
            # x3 = keras.layers.Dropout(0.5)(x3)
            # Skip connection 2.
            encoded_patches = keras.layers.Add()([x3, x2])

            # encoded_patches = trans.TransformerBlock(
            #     num_heads,
            #     embedding_dimensions,
            #     dropout_rate=0.1,
            # )(encoded_patches)

        representation = keras.layers.LayerNormalization(
            epsilon=1e-6)(encoded_patches)
        representation = keras.layers.Flatten()(representation)
        representation = keras.layers.Dropout(0.5)(representation)

        features = trans.mlp(representation, hidden_units=[
                             2048, 1024], dropout_rate=0.5)

        classification = keras.layers.Dense(1)(features)

        return Model(inputs=[input_ab, input_l], outputs=classification)

    def discriminator(self):

        input_ab = Input(shape=self.img_shape_2, name='ab_input')
        input_l = Input(shape=self.img_shape_1, name='l_input')
        net = keras.layers.concatenate([input_l, input_ab])
        net = keras.layers.Conv2D(
            64, (4, 4), padding='same', strides=(2, 2))(net)  # 112, 112, 64
        net = LeakyReLU()(net)
        net = keras.layers.Conv2D(
            128, (4, 4), padding='same', strides=(2, 2))(net)  # 56, 56, 128
        net = LeakyReLU()(net)
        net = keras.layers.Conv2D(
            256, (4, 4), padding='same', strides=(2, 2))(net)  # 28, 28, 256
        net = LeakyReLU()(net)
        net = keras.layers.Conv2D(
            512, (4, 4), padding='same', strides=(1, 1))(net)  # 28, 28, 512
        net = LeakyReLU()(net)
        net = keras.layers.Conv2D(
            1, (4, 4), padding='same', strides=(1, 1))(net)  # 28, 28,1
        return Model([input_ab, input_l], net)

    def colorization_model(self):

        input_img = Input(shape=self.img_shape_3)

        # VGG16 without top layers
        VGG_model = applications.vgg16.VGG16(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model_ = Model(VGG_model.input, VGG_model.layers[-6].output)
        model = model_(input_img)

        # Global Features

        global_features = keras.layers.Conv2D(
            512, (3, 3), padding='same', strides=(2, 2), activation='relu')(model)
        global_features = keras.layers.BatchNormalization()(global_features)
        global_features = keras.layers.Conv2D(
            512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)

        global_features = keras.layers.Conv2D(
            512, (3, 3), padding='same', strides=(2, 2), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)
        global_features = keras.layers.Conv2D(
            512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)

        global_features2 = keras.layers.Flatten()(global_features)
        global_features2 = keras.layers.Dense(1024)(global_features2)
        global_features2 = keras.layers.Dense(512)(global_features2)
        global_features2 = keras.layers.Dense(256)(global_features2)
        global_features2 = keras.layers.RepeatVector(28*28)(global_features2)
        global_features2 = keras.layers.Reshape(
            (28, 28, 256))(global_features2)

        global_featuresClass = keras.layers.Flatten()(global_features)
        global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
        global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
        global_featuresClass = keras.layers.Dense(
            1000, activation='softmax')(global_featuresClass)

        # Midlevel Features

        midlevel_features = keras.layers.Conv2D(
            512, (3, 3),  padding='same', strides=(1, 1), activation='relu')(model)
        midlevel_features = keras.layers.BatchNormalization()(midlevel_features)
        midlevel_features = keras.layers.Conv2D(256, (3, 3),  padding='same', strides=(
            1, 1), activation='relu')(midlevel_features)
        midlevel_features = keras.layers.BatchNormalization()(midlevel_features)

        # fusion of (VGG16 + Midlevel) + (VGG16 + Global)
        modelFusion = keras.layers.concatenate(
            [midlevel_features, global_features2])

        # Fusion + Colorization
        outputModel = keras.layers.Conv2D(
            256, (1, 1), padding='same', strides=(1, 1), activation='relu')(modelFusion)
        outputModel = keras.layers.Conv2D(
            128, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)

        outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
        outputModel = keras.layers.Conv2D(
            64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        outputModel = keras.layers.Conv2D(
            64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)

        outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
        outputModel = keras.layers.Conv2D(
            32, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        outputModel = keras.layers.Conv2D(2, (3, 3), padding='same', strides=(
            1, 1), activation='sigmoid')(outputModel)
        outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
        final_model = Model(inputs=[input_img], outputs=[
                            outputModel, global_featuresClass])

        return final_model

    def train(self, data, test_data, log, sample_interval=1):

        # Create folder to save models if needed.
        save_models_path = os.path.join(config.MODEL_DIR, config.TEST_NAME)
        if not os.path.exists(save_models_path):
            os.makedirs(save_models_path)

        # Load VGG network
        VGG_modelF = applications.vgg16.VGG16(
            weights='imagenet', include_top=True)

        # Real, Fake and Dummy for Discriminator
        positive_y = np.ones((config.BATCH_SIZE, 1), dtype=np.float32)
        negative_y = -positive_y
        # dummy_y = np.zeros((config.BATCH_SIZE, 1), dtype=np.float32)

        # total number of batches in one epoch
        total_batch = int(data.size/config.BATCH_SIZE)

        for epoch in range(config.NUM_EPOCHS):
            for batch in range(total_batch):
                # new batch
                trainL, trainAB, _, original, l_img_oritList = data.generate_batch()
                l_3 = np.tile(trainL, [1, 1, 1, 3])

                # GT vgg
                predictVGG = VGG_modelF.predict(l_3)

                # train generator
                g_loss = self.combined.train_on_batch([l_3, trainL],
                                                      [trainAB, predictVGG, positive_y])
                # train discriminator
                predAB, _ = self.colorizationModel(img_L_3)

                d_loss = self.discriminator_model.train_on_batch(
                    [trainL, trainAB, predAB], [positive_y, negative_y])

                # update log files
                write_log(self.callback, self.train_names,
                          g_loss, (epoch*total_batch+batch+1))
                write_log(self.callback, self.disc_names,
                          d_loss, (epoch*total_batch+batch+1))

                if (batch) % 1 == 0:
                    print("[Epoch %d] [Batch %d/%d] [generator loss: %08f] [discriminator loss: %08f]" %
                          (epoch, batch, total_batch, g_loss[0], d_loss[0]))
            # save models after each epoch
            save_path = os.path.join(
                # save_models_path, "my_model_combinedEpoch%d.h5" % epoch)
                save_models_path, "my_model_combined.h5")
            self.combined.save(save_path)
            # save_path = os.path.join(
            #     save_models_path, "my_model_colorizationEpoch%d.h5" % epoch)
            # self.colorizationModel.save(save_path)
            # save_path = os.path.join(
            #     save_models_path, "my_model_discriminatorEpoch%d.h5" % epoch)
            # self.discriminator.save(save_path)

            # sample images after each epoch
            self.sample_images(test_data, epoch)

    def sample_images(self, test_data, epoch):
        total_batch = int(test_data.size/config.BATCH_SIZE)
        for _ in range(total_batch):
            # load test data
            testL, _,  filelist, original, labimg_oritList = test_data.generate_batch()

            # predict AB channels
            predAB, _ = self.colorizationModel.predict(
                np.tile(testL, [1, 1, 1, 3]))

            # print results
            for i in range(config.BATCH_SIZE):
                originalResult = original[i]
                height, width, channels = originalResult.shape
                predictedAB = cv2.resize(deprocess(predAB[i]), (width, height))
                labimg_ori = np.expand_dims(labimg_oritList[i], axis=2)
                predResult = reconstruct(
                    deprocess(labimg_ori), predictedAB, "epoch"+str(epoch)+"_"+filelist[i][:-5])


if __name__ == '__main__':

    # Create log folder if needed.
    log_path = os.path.join(config.LOG_DIR, config.TEST_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".txt"), "w") as log:
        log.write(str(datetime.datetime.now()) + "\n")

        print('load training data from ' + config.TRAIN_DIR)
        train_data = data.DATA(config.TRAIN_DIR)
        test_data = data.DATA(config.TEST_DIR)
        assert config.BATCH_SIZE <= train_data.size, "The batch size should be smaller or equal to the number of training images --> modify it in config.py"
        print("Train data loaded")

        print("Initiliazing Model...")
        colorizationModel = MODEL()
        print("Model Initialized!")

        print("Start training")
        colorizationModel.train(train_data, test_data, log)
