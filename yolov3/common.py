import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolution(inputs, filters, downsample=False, activate=True, bn=True):
    if downsample:
        # padding=((top_pad, bottom_pad), (left_pad, right_pad))
        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters[-1], kernel_size=filters[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(inputs)
    if bn:
        conv = BatchNormalization()(conv)

    if activate:
        conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(inputs, input_channel, filter_num1, filter_num2):
    short_cut = inputs
    conv = convolution(inputs, filters=(1, 1, input_channel, filter_num1))
    conv = convolution(conv, filters=(3, 3, filter_num1, filter_num2))

    residual_output = short_cut + conv
    return residual_output


def upsample(inputs):
    return tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method='nearest')


def upsample2(inputs, method="deconv"):
    global output
    assert method in ["resize", "deconv"]

    if method == "resize":
        output = tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method='nearest')

    elif method == "deconv":
        filters = inputs.shape.as_list()[-1]
        output = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=(2, 2), padding='same',
                                                 kernel_initializer=tf.random_normal_initializer())(inputs)
    return output
