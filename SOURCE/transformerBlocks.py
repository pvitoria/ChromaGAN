import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import config
import dataClass as data

input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
train_data = data.DATA(config.TRAIN_DIR)


class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, embedding_dimensions, dropout_rate):
        super().__init__()
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dimensions, dropout=0.1
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(2*embedding_dimensions, activation=tf.nn.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(embedding_dimensions, activation=tf.nn.gelu),
                layers.Dropout(dropout_rate),
            ]
        )

    def call(self, patches):
        x1 = self.layer_norm1(patches)
        x2 = self.att(x1, x1)
        x3 = x1 + x2
        x4 = self.layer_norm2(x3)
        x4 = self.ffn(x4)
        return x3+x4


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, embedding_dimensions):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=embedding_dimensions)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embedding_dimensions
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


numclasses = 2
patch_size = 8
num_patches = (config.IMAGE_SIZE // patch_size) ** 2

embedding_dimensions = 64

num_heads = 8

mlp_head_units = [2048, 1024]


def example_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, embedding_dimensions)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(5):
        encoded_patches = TransformerBlock(
            num_heads,
            embedding_dimensions,
            dropout_rate=0.1,
        )(encoded_patches)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units,
                   dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(numclasses)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
