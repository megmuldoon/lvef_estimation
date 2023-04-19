import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from dataloader import EchoMasks

from tensorflow.keras.applications import MobileNetV2

echo_train = EchoMasks(split = 'train' , noise = 0.005, padding=8, batch_size = 1)
echo_valid = EchoMasks(split ='val', padding = 8, batch_size = 3)

class Bottleneck(tf.keras.Model):
    def __init__(self, filters, t, s, out_channels):
        
        super(Bottleneck, self).__init__(name='Bottleneck')

        self.conv2a = tf.keras.layers.Conv2D(t*filters, 1 ,padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.dconv2 = tf.keras.layers.DepthwiseConv2D(3,strides=(s,s),padding ='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(out_channels, 1 ,padding='same')
        self.bn2c = tf.keras.layers.BatchNormalization()

        self.shortcutconv = tf.keras.layers.Conv2D(out_channels, 1,padding='same', strides=(s,s))
        self.shortcutbn = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        shortcut = input_tensor

        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.keras.layers.ReLU(max_value=6)(x)

        x = self.dconv2(x)
        x = self.bn2b(x, training=training)
        x = tf.keras.layers.ReLU(max_value=6)(x)

        x = self.conv2b(x)
        x = self.bn2c(x, training=training)

        shortcut = self.shortcutconv(shortcut)
        shortcut = self.shortcutbn(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


class Residual(tf.keras.Model):
    def __init__(self, filters):
       
        super(Residual, self).__init__(name='')

        self.conv2a = tf.keras.layers.Conv2D(filters, (1, 1),padding='valid')
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.shortcutconv = tf.keras.layers.Conv2D(filters, 1,padding='same', strides=(1,1))
        self.shortcutbn = tf.keras.layers.BatchNormalization()


    def call(self, input_tensor, training=False):
        shortcut = input_tensor

        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        shortcut = self.shortcutconv(shortcut)
        shortcut = self.shortcutbn(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


class Upsample(tf.keras.Model):
	def __init__(self, filters):
		super(Upsample, self).__init__(name='')
		self.filters = filters
		self.up_samp = layers.UpSampling2D((2,2))

		self.conv1 = layers.Conv2D(filters, (3, 3), padding="same")
		self.bn1 = layers.BatchNormalization()

		self.conv2 = layers.Conv2D(filters, (3, 3), padding="same")
		self.bn2 = layers.BatchNormalization()

	def call(self, inputs, conv_features):
		# upsample
   		x = self.up_samp(inputs)
   		x = layers.Concatenate()([x, conv_features])

   		x = self.conv1(x)
   		x = self.bn1(x)
   		x = tf.nn.relu(x)

   		x = self.conv2(x)
   		x = self.bn2(x)
   		x = tf.nn.relu(x)

   		return(x)


class RMUNet(tf.keras.Model):
	def __init__(self):

		super(RMUNet,self).__init__(name='RMUNet')
		self.conv1 = layers.Conv2D(32,3, (2,2), padding='same')
		
		self.b1 = Bottleneck(filters= 32, s = 1, t=1, out_channels=16)
		self.b2 = Bottleneck(filters= 16, s = 2, t=6, out_channels=24)
		self.r1 = Residual(24)
		self.b3 = Bottleneck(filters= 24, s = 2, t=6, out_channels=32)
		self.r2 = Residual(32)
		self.b4 = Bottleneck(filters= 32, s = 2, t=6, out_channels=64)
		self.r3 = Residual(64)
		self.b5 = Bottleneck(filters=64, s = 1, t=6, out_channels=96)
		self.b6 = Bottleneck(filters=96, s=2, t=6, out_channels=160)
		self.b7 = Bottleneck(filters=160, s=1, t=6, out_channels=320)

		self.conv2 = layers.Conv2D(1280,1, padding='same')

		self.us1 = Upsample(filters=512)
		self.us2 = Upsample(filters=256)
		self.us3 = Upsample(filters=128)
		self.us4 = Upsample(filters=64)

		self.conv3 = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")



	def call(self, inputs, training=False):
		x = self.conv1(inputs)
		s4 = self.b1(x)
		s3 = self.b2(s4)
		x = self.r1(s3)
		x = self.b3(x)
		s2 = self.r2(x)
		x = self.b4(s2)
		x = self.r3(x)
		s1 = self.b5(x)
		x = self.b6(s1)
		x = self.b7(x)

		x = self.conv2(x)

		x = self.us1(x, s1)
		x = self.us2(x, s2)
		x = self.us3(x, s3)
		x = self.us4(x, s4)
		
		x = layers.UpSampling2D((2,2))(x)
		x = self.conv3(x)

		return x
	


RMUnet = RMUNet()	
def display(display_list):
  	plt.figure()

  	title = ['Input Image', 'True Mask', 'Predicted Mask']

  	for i in range(len(display_list)):
  		plt.subplot(1, len(display_list), i+1)
  		plt.title(title[i])
  		plt.imshow(display_list[i])
  		plt.axis('off')
  	plt.show()

class DisplayCallback(tf.keras.callbacks.Callback):
	def __init__(self):
		super(DisplayCallback,self).__init__()

	def on_epoch_end(self, epoch, logs=None):
		show_predictions()

def show_predictions():
	sample_img, sample_mask = echo_train.__getitem__(4)
	pred_mask = RMUnet.predict(sample_img)
	display([sample_img[0], sample_mask[0], pred_mask[0]])


# sample_img, sample_mask = echo_train.__getitem__(4)
# print(np.shape(sample_img[0]), "  ", np.shape(sample_mask))


# RMUnet.compile(optimizer=tf.keras.optimizers.Adam(),
#                   loss='binary_crossentropy',
#                   metrics="accuracy")

# pred_mask = RMUnet.predict(sample_img)
# display([sample_img[0], sample_mask[0], pred_mask[0]])

# callbacks = callbacks = [tf.keras.callbacks.ModelCheckpoint("RMU", save_best_only=True, save_weights_only=True), DisplayCallback()]

# model_history = RMUnet.fit(echo_train,epochs=10,validation_data=echo_valid, callbacks=callbacks)








