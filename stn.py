import numpy as np
import tensorflow as tf
import keras
from keras.layers import *


class BilinearInterpolation(Layer):
    def build(self, input_shape):
        assert len(input_shape) == 2, "Expected exactly two inputs"
        assert input_shape[1][-1] == 6, "Wrong affine parameters"
        self._out_shape = self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(np.squeeze(input_shape[0]))

    def call(self, tensors, mask=None):
        X, transformation = tensors
        Y = self._transform(X, transformation)
        return tf.reshape(Y, [-1, *self._out_shape[1:]])

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype="int32")
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1.0, 1.0, width)
        y_linspace = tf.linspace(-1.0, 1.0, height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, shape=(1, -1))
        y_coordinates = tf.reshape(y_coordinates, shape=(1, -1))
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, locnet_x, locnet_y):
        output_size = tf.shape(locnet_x)[1:]
        batch_size = tf.shape(locnet_x)[0]
        height = tf.shape(locnet_x)[1]
        width = tf.shape(locnet_x)[2]
        num_channels = tf.shape(locnet_x)[3]

        locnet_y = tf.reshape(locnet_y, shape=(batch_size, 2, 3))

        locnet_y = tf.reshape(locnet_y, (-1, 2, 3))
        locnet_y = tf.cast(locnet_y, "float32")

        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1])  # flatten?
        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, tf.stack([batch_size, 3, -1]))

        transformed_grid = tf.matmul(locnet_y, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x = tf.reshape(x_s, [-1])
        y = tf.reshape(y_s, [-1])

        # Interpolate
        height_float = tf.cast(height, dtype="float32")
        width_float = tf.cast(width, dtype="float32")

        output_height = output_size[0]
        output_width = output_size[1]

        x = tf.cast(x, dtype="float32")
        y = tf.cast(y, dtype="float32")
        x = 0.5 * (x + 1.0) * width_float
        y = 0.5 * (y + 1.0) * height_float

        x0 = tf.cast(tf.floor(x), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), "int32")
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype="int32")
        max_x = tf.cast(width - 1, dtype="int32")
        zero = tf.zeros([], dtype="int32")

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width * height
        pixels_batch = tf.range(batch_size) * flat_image_dimensions
        flat_output_dimensions = output_height * output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(locnet_x, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype="float32")
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, "float32")
        x1 = tf.cast(x1, "float32")
        y0 = tf.cast(y0, "float32")
        y1 = tf.cast(y1, "float32")

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        transformed_image = tf.add_n(
            [
                area_a * pixel_values_a,
                area_b * pixel_values_b,
                area_c * pixel_values_c,
                area_d * pixel_values_d,
            ]
        )
        return tf.reshape(
            transformed_image,
            shape=(batch_size, output_height, output_width, num_channels),
        )


class SpatialTransformer(keras.models.Model):
    def __init__(self, localization=[], **kwargs):
        super().__init__(**kwargs)
        self.localization = localization
        self.locnet_flatten = Flatten()
        self.locnet_dense_1 = Dense(64, activation="relu")
        self.locnet_dense_2 = Dense(
            6, weights=[tf.zeros((64, 6)), np.float32([[1, 0, 0], [0, 1, 0]]).flatten()]
        )
        self.interpolation = BilinearInterpolation()

    def call(self, inputs):
        x = inputs
        for layer in self.localization:
            x = layer(x)
        locnet = self.locnet_flatten(x)
        locnet = self.locnet_dense_1(locnet)
        locnet = self.locnet_dense_2(locnet)
        return self.interpolation([inputs, locnet])
