import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 8), pool_size=(1, 2)):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.activation = tf.keras.layers.LeakyReLU(negative_slope=0.01)
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_size)

    def call(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return self.pool(x)


class SpectralCNN(tf.keras.Model):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.block1 = ConvBlock(64)
        self.block2 = ConvBlock(64)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128)
        self.activation = tf.keras.layers.LeakyReLU(negative_slope=0.01)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)
