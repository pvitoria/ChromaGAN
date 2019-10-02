

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



GRADIENT_PENALTY_WEIGHT = 10  # As per the paper



def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)



def reconstruct(batchX, predictedY, filelist, epoch, i):
    #for i in range(config.BATCH_SIZE):
    result = np.concatenate((batchX[i], predictedY[i]), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    save_path = os.path.join(config.OUT_DIR, "ChromaGAN_results/"+ filelist[i][:-4] +  "Epoch%dreconstructed.jpg" % epoch)
    cv2.imwrite(save_path, result)
    return result

def reconstruct_no(batchX, predictedY, filelist, epoch, i):
    #for i in range(config.BATCH_SIZE):
    result = np.concatenate((batchX[i], predictedY[i]), axis=2)
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
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

    
class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

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


# The gradient penalty loss function requires the input averaged samples to get
# gradients. However, Keras loss functions can only have two arguments, y_true and
# y_pred. We get around this by making a partial() of the function with the averaged
        # samples here.
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
# Functions need names or Keras will throw an error
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
        self.val_names = ['val_loss', 'val_mae']



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
        save_models_path = os.path.join(config.OUT_DIR,config.TEST_NAME)
        if not os.path.exists(save_models_path):
                os.makedirs(save_models_path)
        #save_path = os.path.join(config.OUT_DIR, "test33_/my_model_colorization33Epoch0.h5")
        #self.colorizationModel =load_model(save_path)  # creates a HDF5 file 'my_model.h5'
        #save_path = os.path.join(config.OUT_DIR, "test33_/my_model_discriminator33Epoch0.h5")
        #self.discriminator = model_from_json(open(save_path).read())
        #self.discriminator.load_weights(save_path)
        #self.discriminator =load_model(save_path, custom_objects={'wasserstein_loss': wasserstein_loss})  # creates a HDF5 file 'my_model.h5'
        #save_path = os.path.join(config.OUT_DIR, "test33_/my_model_combined33Epoch0.h5")
        #self.combined =load_model(save_path , custom_objects={'wasserstein_loss': wasserstein_loss})  # creates a HDF5 file 'my_model.h5'
        for epoch in range(config.NUM_EPOCHS):
                for batch in range(int(data.size/config.BATCH_SIZE)):
                    batchX, batchY, _ = data.generate_batch()
                    l_3=np.tile(batchX,[1,1,1,3])
                    predictVGG =VGG_modelF.predict(l_3)
                    g_loss_col =self.combined.train_on_batch([l_3, batchX],
                                                        [batchY, predictVGG, positive_y])
                    write_log(self.callback, self.train_names, g_loss_col, it)
                    it = it+1
                    g_loss =  g_loss_col[0] #+ g_loss_disc[0]+ g_loss_col[2])
                    ##original = np.concatenate((batchX, batchY), axis=3)
                    #fake_ab, _ = self.colorizationModel.predict(np.tile(batchX,[1,1,1,3]))
                    d_loss_real = self.discriminator_model.train_on_batch([batchX, batchY, l_3], [positive_y, negative_y, dummy_y])


                    #d_loss_real = self.discriminator.train_on_batch([batchY, batchX], valid)
                    #d_loss_fake = self.discriminator.train_on_batch([fake_ab, batchX], fake)
                    #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    if (batch+1)%1000 ==0: 
                        print("[Epoch %d] [Batch %d/%d] [loss: %08f]" %  ( epoch, batch,total_batch, g_loss_col[0]))
                        save_path = os.path.join(save_models_path, "my_model_combined33Epoch%d_it%d.h5" % (epoch, it))
                        self.combined.save(save_path)  # creates a HDF5 file 'my_model.h5'
                        save_path = os.path.join(save_models_path, "my_model_colorization33Epoch%d_it%d.h5" % (epoch, it))
                        self.colorizationModel.save(save_path)  # creates a HDF5 file 'my_model.h5'
                        save_path = os.path.join(save_models_path, "my_model_discriminator33Epoch%d_it%d.h5" % (epoch, it))
                        self.discriminator.save_weights(save_path)  # creates a HDF5 file 'my_model.h5'
                        save_path = os.path.join(save_models_path, "my_model_combined33Epoch%d_it%dWeihts.h5" % (epoch, it))
                        self.combined.save(save_path)  # creates a HDF5 file 'my_model.h5'
                        save_path = os.path.join(save_models_path, "my_model_colorization33Epoch%d_it%dWeihts.h5" % (epoch, it))
                        self.colorizationModel.save(save_path)  # creates a HDF5 file 'my_model.h5'
                        save_path = os.path.join(save_models_path, "my_model_discriminator33Epoch%d_it%dWeihts.h5" % (epoch, it))
                        self.discriminator.save_weights(save_path)  # creates a HDF5 file 'my_model.h5'

                save_path = os.path.join(save_models_path, "my_model_combined33Epoch%d.h5" % epoch)
                self.combined.save(save_path)  # creates a HDF5 file 'my_model.h5'
                save_path = os.path.join(save_models_path, "my_model_colorization33Epoch%d.h5" % epoch)
                self.colorizationModel.save(save_path)  # creates a HDF5 file 'my_model.h5'
                save_path = os.path.join(save_models_path, "my_model_discriminator33Epoch%d.h5" % epoch)
                self.discriminator.save(save_path)  # creates a HDF5 file 'my_model.h5'

                save_path = os.path.join(save_models_path, "my_model_combined33Epoch%dWeihts.h5" % epoch)
                self.combined.save_weights(save_path)  # creates a HDF5 file 'my_model.h5'
                save_path = os.path.join(save_models_path, "my_model_colorization33Epoch%dWeihts.h5" % epoch)
                self.colorizationModel.save_weights(save_path)  # creates a HDF5 file 'my_model.h5'
                save_path = os.path.join(save_models_path, "my_model_discriminator33Epoch%dWeihts.h5" % epoch)
                self.discriminator.save_weights(save_path)  # creates a HDF5 file 'my_model.h5'
                #self.sample_images(epoch, batch)


    def sample_images(self, epoch):
        VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True) # none, 1000
        r, c = 3, 1
        avg_cost = 0
        avg_cost2 = 0
        avg_cost3 = 0
        avg_ssim = 0
        avg_psnr = 0
        save_path = os.path.join(config.OUT_DIR, "test33_final/my_model_colorization33Epoch2Weihts.h5")
        self.colorizationModel.load_weights(save_path)
        test_data = data.DATA(config.TEST_DIR)
        total_batch = int(test_data.size/config.BATCH_SIZE)
        print(test_data.size)
        print(total_batch)
        it = 0
        for _ in range(total_batch):
                batchX, batchY,  filelist  = test_data.generate_batch()
                predY, _  = self.colorizationModel.predict(np.tile(batchX,[1,1,1,3]))
                predictVGG =VGG_modelF.predict(np.tile(batchX,[1,1,1,3]))
                loss = self.colorizationModel.evaluate(np.tile(batchX,[1,1,1,3]), [batchY, predictVGG], verbose=0)
                avg_cost += loss[0]/total_batch
                avg_cost2 += loss[1]/total_batch
                avg_cost3 += loss[2]/total_batch
                for i in range(config.BATCH_SIZE):
                    predResult = reconstruct(deprocess(batchX), deprocess(predY), filelist, epoch, i)
                    originalResult = reconstruct_no(deprocess(batchX), deprocess(batchY), filelist, 1000, i)
                    avg_ssim += tf.keras.backend.eval( tf.image.ssim(tf.convert_to_tensor(originalResult, dtype=tf.float32), tf.convert_to_tensor(predResult, dtype=tf.float32), max_val=255))/test_data.size 
                    avg_psnr += tf.keras.backend.eval( tf.image.psnr(tf.convert_to_tensor(originalResult, dtype=tf.float32), tf.convert_to_tensor(predResult, dtype=tf.float32), max_val=255))/test_data.size
             #   self.test_loss_array.append(loss[1])  
                    it = it+1
                    if it%50 ==0: 

                        print(it)
                        print(" ----------  loss =", "{:.8f}------------------".format(avg_cost))
                        print(" ----------  upsamplingloss =", "{:.8f}------------------".format(avg_cost2))
                        print(" ----------  classification_loss =", "{:.8f}------------------".format(avg_cost3))
                        print(" ----------  ssim loss =", "{:.8f}------------------".format(avg_ssim))
                        print(" ----------  psnr loss =", "{:.8f}------------------".format(avg_psnr))

        print(" ----------  loss =", "{:.8f}------------------".format(avg_cost))
        print(" ----------  upsamplingloss =", "{:.8f}------------------".format(avg_cost2))
        print(" ----------  classification_loss =", "{:.8f}------------------".format(avg_cost3))
        print(" ----------  ssim loss =", "{:.8f}------------------".format(avg_ssim))
        print(" ----------  psnr loss =", "{:.8f}------------------".format(avg_psnr))



if __name__ == '__main__':
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    with open(os.path.join(config.LOG_DIR, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".txt"), "w") as log:
        log.write(str(datetime.datetime.now()) + "\n")
        log.write("Use Pretrained Weights: " + str(config.USE_PRETRAINED) + "\n")
        print(config.BATCH_SIZE)
        print("Use Pretrained Weights")
        log.write("Pretrained Model: " + config.PRETRAINED + "\n")
        print("Pretrained Model")
        train_data = data.DATA(config.TRAIN_DIR)
        print("train data")
        colorizationModel = MODEL()
        print("Model Initialized")
        colorizationModel.train(train_data, log)
        #colorizationModel.sample_images(4)
