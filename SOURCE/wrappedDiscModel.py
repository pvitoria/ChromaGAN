import tensorflow as tf
import keras as keras
import config as config

GRADIENT_PENALTY_WEIGHT = 10


class RandomWeightedAverage(keras.layers.merge._Merge):
    def _merge_function(self, inputs):
        weights = tf.random.uniform((config.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


class WrappedDiscriminatorModel(keras.Model):
    def __init__(
        self,
        discriminator,
        colourer,
        **kwargs,
    ):
        inputs = kwargs.get("inputs", None)
        outputs = kwargs.get("outputs", None)
        super().__init__(inputs=inputs, outputs=outputs)
        self.discriminator = discriminator
        self.colourer = colourer

    def compile(self, optimizer):
        super().compile(optimizer)
        self.optimizer = optimizer

    def gradient_penalty(self, img_ab_real, predAB, img_L, gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT):
        averaged_samples = RandomWeightedAverage()([img_ab_real,
                                                    predAB])

        with tf.GradientTape() as tape:
            tape.watch(averaged_samples)
            pred = self.discriminator([averaged_samples, img_L])

        grads = gp_tape.gradient(pred, [interpolated])[0]

        grads_sqr = tf.square(grads)
        grads_sqr_sum = tf.reduce_sum(gradients_sqr,
                                      axis=np.arange(1, len(grads_sqr.shape)))
        grad_l2_norm = tf.sqrt(gradients_sqr_sum)
        gradient_penalty = gradient_penalty_weight * \
            tf.square(1 - grad_l2_norm)
        return tf.reduce_mean(gradient_penalty)

    def train_step(self, data):
        (x, y) = data
        (img_L, img_ab_real, img_l3) = x
        (positive_y, negative_y) = y

        img_ab_fake, _ = self.colourer(img_l3)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            (real_pred, fake_pred) = y_pred

            d_loss = -wasserstein_loss(positive_y, real_pred)
            d_loss += wasserstein_loss(negative_y, fake_pred)
            d_loss += self.gradient_penalty(img_ab_real, img_ab_fake, img_L)

            d_loss = gp

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(
            d_loss, self.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.trainable_variables)
        )
        return {"loss": d_loss}
