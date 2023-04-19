from data_loader import EchoMasks
from uNet import build_unet_model
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import schedules
import numpy as np
import matplotlib.pyplot as plt
from RMUNet import RMUNet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



echo_train = EchoMasks(split = 'train' , padding=8, batch_size = 1)
echo_valid = EchoMasks(split ='val', padding = 8, batch_size = 1)
echo_test = EchoMasks(split ='test', padding = 8, batch_size = 1)

model = RMUNet()


def display(display_list, save_as=None):
	plt.figure()
	title = ['Input Image', 'Prev Mask', 'True Mask', 'Predicted Mask']
	
	for i in range(len(display_list)):
		plt.subplot(1, len(display_list), i+1)
		plt.title(title[i])
		plt.imshow(display_list[i])
		plt.axis('off')

	if save_as:
		plt.savefig('Prediction_epoch' + str(save_as))

	else:
		plt.show()

def diceLoss(y_true, y_pred, smooth=1e-6):
	#flatten label and prediction tensors
	intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
	union = keras.backend.sum(y_true, axis=[1,2,3]) +keras.backend.sum(y_pred, axis=[1,2,3])
	dice = keras.backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
	return 1-dice

class DisplayCallback(tf.keras.callbacks.Callback):
	def __init__(self):
		super(DisplayCallback,self).__init__()

	def on_epoch_end(self, epoch, logs=None):
		show_predictions(title=str(epoch))

def show_predictions(title=None):
	sample_img, sample_mask = echo_train.__getitem__(4)

	pred_mask = model.predict(sample_img)
	display([sample_img[2,:,:, 0:3],sample_img[2,:, :,3], sample_mask[2], pred_mask[2]], save_as = title)

sample_img, sample_mask = echo_test.__getitem__(4)

lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=20000,
    decay_rate=0.9)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss= diceLoss,
                  metrics="accuracy")

#See what the model predicts before training 
pred_mask = model.predict(sample_img)

model.summary()
# Do not save as .h5 file
callbacks = [tf.keras.callbacks.ModelCheckpoint("RMU", save_best_only=True, save_weights_only=True), DisplayCallback()]
display([sample_img[2, :, :, 0:3], sample_img[2,:, :,3],sample_mask[2], pred_mask[2]])

print(np.shape(sample_img[2, :, :, 3]))

model_history = model.fit(echo_train,epochs=30,validation_data=echo_test, callbacks=callbacks)
