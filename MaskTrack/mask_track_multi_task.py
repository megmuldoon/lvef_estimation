from data_loader import EchoSet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from RMUNet import RMUNet


import cv2
import imageio
import os
from skimage.morphology import dilation

bce_metric = keras.metrics.BinaryCrossentropy(name="bce")
acc = keras.metrics.BinaryAccuracy(name="seg_acc")
loss_tracker = keras.metrics.Mean(name="loss")

# echo_train = EchoSet(split = 'train' , pad=8, batch_size = 1)
# echo_valid = EchoSet(split ='val', pad = 8, batch_size = 1)
#echo_test = EchoSet(split ='test', pad = 8, batch_size = 1, center=False)

# frame0_net = RMUNet()
# frame0_net.load_weights("RMU_3channel")

# data = echo_test.__getitem__(2)

# X, y = echo_test.__getitem__(12)
# print("True LVEF: ", y.ef)

# print("Input shape: ", np.shape(X))
# print("Video masks shape", np.shape(y.video_masks))
# print("Ed_es_frames shape", np.shape(y.ed_es_frames))


def dice_coef(y_true, y_pred, smooth=0.00001):
    y_pred = tf.keras.layers.ThresholdedReLU(0.5)(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.backend.sum(y_true, axis=[1,2,3]) +keras.backend.sum(y_pred, axis=[1,2,3])
    dice = keras.backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

class reshapeLayer(tf.keras.layers.Layer):
    def __init__(self,shape, name):
        super().__init__(name=name)
        self.shape = shape

    def call(self, inputs):
        return tf.reshape(tf.convert_to_tensor(inputs), self.shape)

class multi_task(keras.Model):
    def __init__(self, bound_loss=True):
        super(multi_task, self).__init__(name='multi_task')
        self.MAX_FRAMES = 64
        self.bound_loss = bound_loss
        # Establish the maskTrack network, want this to be trainable 
        self.maskTrack = RMUNet()
        self.maskTrack.load_weights("RMUNet_augs2").expect_partial() #expect_partial to silence checkpoint restore warnings

        # to reshape the mask returned from MaskTrack framework 
        self.rsp_mask = reshapeLayer((128,128,1), name='rsp1')

        self.frame0_net = RMUNet()
        self.frame0_net.load_weights("RMU_3channel").expect_partial()



    def call(self, inputs, training=False):
        # put the first frame through the three channel U-Net to get the mask
        footprint = np.array([[0,0,1,0,0], [0,1,1,1,0], [1,1,1,1,1], [0,1,1,1,0], [0,0,1,0,0]])
        f0 = inputs[:, 0, :]

        # Append a mask of zeros to the first frame
        # f0 = tf.concat([f0, np.zeros((1, 128, 128,1), dtype=np.float32)], axis=-1)


        masks = []
        # prev_mask = self.maskTrack(f0)
        prev_mask = self.frame0_net(f0)

        final_mask = self.rsp_mask(prev_mask)
        masks.append(final_mask)

        prev_mask = np.where(prev_mask < 0.5, 0.0, 1.0)
        coarsened_mask = dilation(prev_mask[0,:,:,0], footprint)

        coarsened_mask = np.expand_dims(coarsened_mask, axis=-1)
        coarsened_mask = np.expand_dims(coarsened_mask, axis=0)


        #masks = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # print(np.shape(inputs))
        # print(np.shape(inputs)[1])

        for f in range(1,np.shape(inputs)[1]):
            frame = inputs[:, f, :] 
            # Append the prev mask with the current frame 
            frame_and_mask = tf.concat([frame, coarsened_mask], axis=-1) # shape = (batch, 128, 128, 4)
            
            prev_mask = self.maskTrack(frame_and_mask)

            #Remove the batch axis to stack the final masks (so that masks shape = (MAX-1, 128, 128, 1))
            final_mask = self.rsp_mask(prev_mask)
            #masks = masks.write(masks.size(), final_mask)
            masks.append(final_mask)

            prev_mask = np.where(prev_mask < 0.5, 0.0, 1.0)
            coarsened_mask = dilation(prev_mask[0,:,:,0], footprint)

            coarsened_mask = np.expand_dims(coarsened_mask, axis=-1)
            coarsened_mask = np.expand_dims(coarsened_mask, axis=0)

            
        #print("mask shape:", np.shape(masks))
        outputs = tf.stack(masks) 
        outputs = tf.expand_dims(outputs, 0) # Return shape = (1, MAX-1, 128, 128, 1)
        #print("output shape:", np.shape(outputs))
        return outputs

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            
            # y_pred is the segmentation mask for the entire video
            seg_pred = self(x, training=True) 
            seg_pred = tf.boolean_mask(seg_pred, y.ed_es_frames)

            seg_ground_truth = tf.boolean_mask(y.video_masks, y.ed_es_frames)
            bce = tf.keras.losses.BinaryCrossentropy()
            seg_pred = np.where(seg_pred > 0.5, 1.0, 0.0)
            dice = dice_coef(seg_ground_truth, seg_pred)

            loss = bce(seg_pred, seg_ground_truth)
    
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(dice)
        acc.update_state(seg_pred, seg_ground_truth)

        return{m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
       
        # y_pred is the segmentation mask for the entire video
        seg_pred = self(x, training=True) 
        seg_pred = tf.boolean_mask(seg_pred, y.ed_es_frames)
        seg_ground_truth = tf.boolean_mask(y.video_masks, y.ed_es_frames)
        seg_pred = tf.experimental.numpy.where(seg_pred > 0.5, 1.0, 0.0)
        dice = dice_coef(seg_ground_truth, seg_pred)

        loss_tracker.update_state(dice)
        acc.update_state(seg_pred, seg_ground_truth)

        return{m.name: m.result() for m in self.metrics}

    def make(self, input_shape=(248,128,128,3)):
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='actor')
        print(model.summary())
        return model
    @property
    def metrics(self):
        return [loss_tracker, acc]

# model = multi_task()
# model.compile(optimizer=tf.keras.optimizers.Adam())

# for item_idx in range(0, echo_test.__len__()):
#     X, y = echo_test.__getitem__(item_idx)
#     outs = model.call(X)
#     print(np.shape(outs), "     ", np.shape(y.ed_es_frames))
#     seg_pred = tf.boolean_mask(outs[0], y.ed_es_frames[1:])
#     seg_ground_truth = tf.boolean_mask(y.video_masks, y.ed_es_frames)
#     seg_pred = tf.experimental.numpy.where(seg_pred > 0.5, 1.0, 0.0)
#     dice = dice_coef(seg_ground_truth, seg_pred)
#     print(dice)
    
#model.evaluate(echo_test)

# outs = model.call(X)
# seg_pred_avg = np.where(outs[0] > 0.45, 1, 0)
# print(np.shape(seg_pred_avg))
# print(np.shape(outs))
# seg_pred = tf.boolean_mask(outs, y.ed_es_frames)
# print(np.shape(seg_pred))


# outs_flipped = model.call(np.flip(X, axis = 1))
# outs_flipped = np.flip(outs_flipped, axis=1)
# seg_pred_avg = (outs[0] + outs_flipped[0]) / 2
# seg_pred_avg = np.where(seg_pred_avg > 0.45, 1.0, 0.0)
#seg_pred_flipped = tf.boolean_mask(outs_flipped, y.ed_es_frames)

# seg_ground_truth = tf.boolean_mask(y.video_masks, y.ed_es_frames)
# print(np.shape(seg_ground_truth))

# true_areas = tf.math.reduce_sum(seg_ground_truth, axis =(1, 2))
# print(true_areas)

#areas = tf.math.reduce_sum(outs, axis =(2, 3))
# print(np.shape(areas))
# plt.scatter(range(0, 63), areas[0, :,0])
# plt.show()


# bce = tf.keras.losses.BinaryCrossentropy()
# loss = bce(seg_pred, seg_ground_truth)

# print("Forward loss: ", loss)
# loss = bce(seg_pred_flipped, seg_ground_truth)
# print("Backwards loss: ", loss)


# seg_pred_avg = (seg_pred + seg_pred_flipped) / 2
# seg_pred_avg = np.where(seg_pred_avg > 0.45, 1.0, 0.0)
# loss = bce(seg_pred_avg, seg_ground_truth)
# print("Average loss: ", loss)

# fig, axs = plt.subplots(4, 2, sharey=True, tight_layout=True)
# axs[0][0].imshow(seg_ground_truth[0])
# axs[0][1].imshow(seg_ground_truth[1])
# axs[1][0].imshow(seg_pred[0])
# axs[1][1].imshow(seg_pred[1])
# axs[2][0].imshow(seg_pred_flipped[0])
# axs[2][1].imshow(seg_pred_flipped[1])
# axs[3][0].imshow(seg_pred_avg[0])
# axs[3][1].imshow(seg_pred_avg[1])
 

# plt.show()
############## Simpsons Biplane Estimation #######
# areas = areas[0].numpy()
# max_frame = np.argmax(areas)
# min_frame = np.argmin(areas)

# print("(Using only mask area in pixels) Max frame: ", max_frame, ", Min frame: ", min_frame)
# print("Labeled frames: ", np.nonzero(y.ed_es_frames[0,:]))

# true_max_frame = np.nonzero(y.ed_es_frames[0,:])[0][0]
# true_min_frame = np.nonzero(y.ed_es_frames[0,:])[0][1]
# print("True max frame: ", true_max_frame, ", True min frame: ", true_min_frame)

# test = seg_pred_avg[max_frame]
# test = test.astype(np.uint8)

# contours, _ = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]


# rect = cv2.minAreaRect(cnt)
# width = int(rect[1][0])
# height = int(rect[1][1])

# print(areas[max_frame], "   ", height)

# V_max = (8 * (areas[max_frame])**2) / (3 * np.pi * height)
# i = 0
# gif_names = []
# vols = np.zeros(np.shape(areas)[0])
# for frame in seg_pred_avg:
    
#     frame = frame.astype(np.uint8)

#     # Get the contour of the LV
#     contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cnt = contours[0]

#     # Find the minumum enclosing triangle for the LV contour
#     [area, tri] = cv2.minEnclosingTriangle(cnt)

#     # Find the three points closest to the triangle corners
#     idx, arr = closest_point(tri[0][0],cnt[:,0,:])
#     bp1 = cnt[idx, 0, :]
#     idx, arr = closest_point(tri[1][0],cnt[:,0,:])
#     bp2 = cnt[idx, 0, :]
#     idx, arr = closest_point(tri[2][0],cnt[:,0,:])
#     bp3 = cnt[idx, 0, :]

#     # one of these points is the apex and the other two indicate the MV annulus 
#     apex, mv_1, mv_2 = label_points(bp1, bp2, bp3)

#     # Find the slope of the line connecting the MV points
#     mv_slope = mv_1 - mv_2 

#     # PERPINDICULAR BISECTOR
#     mid_point = ((mv_1 + mv_2) /2).astype(int)

#     # The apex slope should be perpindicular to the mv_slope
#     apex_slope = apex - mid_point
#     apex_slope = apex_slope / np.linalg.norm(apex_slope)

#     blank = np.zeros(np.shape(frame))
#     apex_line = cv2.line(blank.copy(), (apex - 112 * apex_slope).astype(int), (mid_point + 112 * apex_slope).astype(int), 1, 1)
#     contour_drawing = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
#     intersections = np.logical_and(contour_drawing, apex_line)
#     points = np.where(intersections == 1)
#     points = np.transpose(points[0:2]) # put points array in the form [[x1, y1], [x2, y2] ...]

#     # error handling (ensure there are at least 2 intercept pts, which are far from apex)
#     width = 2
#     idx,arr = farthest_point(np.flip(apex), points)
#     dist = arr[idx]
  

#     while np.shape(points)[0] < 2 or dist < 250:
#         apex_line = cv2.line(apex_line, (apex + 10 * apex_slope).astype(int), (mid_point - 10 * apex_slope).astype(int), 1, width)
#         contour_drawing = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
#         intersections = np.logical_and(contour_drawing, apex_line)
#         points = np.where(intersections == 1)
#         points = np.transpose(points[0:2])
#         idx,arr = farthest_point(np.flip(apex), points)
#         dist = arr[idx]

#         width = width + 1

#     # Choose the point that is furthest away from the apex as the intercept.  
#     intercept = points[idx]

    

    # New apex is the interception point furthest from the midpoint 
    # idx, arr = farthest_point(intercept, points)
    # apex = points[idx]

    # intercept = np.flip(intercept)
    
    # # LV height is the length of the line connecting the intercept and the apex
    # LV_length = np.linalg.norm(apex - intercept)

    # V = (8 * (areas[i])**2) / (3 * np.pi * LV_length)
    # print(i, ": ", V)
    # vols[i] = V


    # tri = tri.astype(int)
    # p1 = (tri[0][0][0], tri[0][0][1])
    # p2 = (tri[1][0][0], tri[1][0][1])
    # p3 = (tri[2][0][0], tri[2][0][1])

    # im3 = X[0][i].copy() * 255

    # im3 = cv2.drawContours(im3, [cnt], 0, (255,255,0), 1)
    #im3 = cv2.line(im3, (apex - 112 * apex_slope).astype(int), (mid_point + 112 * apex_slope).astype(int), (0,255,0), 1)
    # im3 = cv2.line(im3, p1, p2, (0,255,0), 1)
    # im3 = cv2.line(im3, p3, p2, (0,255,0), 1)
    # im3 = cv2.line(im3, p1, p3, (0,255,0), 1)

    # im3 = cv2.circle(im3, apex, 2, (0, 0, 255), 2)
    # im3 = cv2.circle(im3, mv_1, 2, (255, 0, 0), 2)
    # im3 = cv2.circle(im3, mv_2, 2, (255, 0, 0), 2)
    # im3 = cv2.circle(im3, intercept, 2, (0,0,255), 2)

    # im3 = cv2.line(im3, bp1, bp3, (0, 255, 255), 1)
    # im3 = cv2.line(im3, bp1, bp2, (0, 255, 255), 1)
    # im3 = cv2.line(im3, bp2, bp3, (0, 255, 255), 1)

#     im3 = cv2.line(im3, apex, intercept, (255, 0,255), 1)

#     filename = 'gif_images/f' + str(i) + '.png'
#     gif_names.append(filename)
#     cv2.imwrite(filename, im3)
#     # cv2.imshow('image',im3)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     i = i +1

# with imageio.get_writer('measure12_mv2apex.gif', mode='I') as writer:
#     for filename in gif_names:
#         image = imageio.imread(filename)
#         writer.append_data(image)

# #Remove files
# for filename in set(gif_names):
#     os.remove(filename)


###########################################

# max_frame = np.argmax(vols)
# min_frame = np.argmin(vols)
# print("(Using estimate LV volume) Max frame: ", max_frame, ", Min frame: ", min_frame)

# EF = (vols[max_frame] - vols[min_frame]) / vols[max_frame]

# print(EF)

# plt.scatter(range(0, 31), vols)
# plt.show()



#model_history = model.fit(echo_train,epochs=10,validation_data=echo_valid)





