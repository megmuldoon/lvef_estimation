from dataloader import EchoSet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from RMUNet import RMUNet

import cv2
import imageio
import os

import scipy.signal as sig
import scipy.stats as stat
import scipy.spatial.distance as sdist
import matplotlib.colors as clr

def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance

def farthest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmax(distance), distance

def label_points(p1, p2, p3):
    d1 = np.linalg.norm(p1 - p2)
    d2 = np.linalg.norm(p2 - p3)
    d3 = np.linalg.norm(p1 - p3)
    if d1 < d2 and d1 < d3:
        apex = p3
        mv_1 = p1
        mv_2 = p2
    elif d2 < d1 and d2 < d3:
        apex = p1
        mv_1 = p3
        mv_2 = p2
    else:
        apex = p2
        mv_1 = p1
        mv_2 = p3

    return apex, mv_1, mv_2


def simpsons_biplane(cnt, frame, mask_area,draw, im):
    # Find the minumum enclosing triangle for the LV contour
    [area, tri] = cv2.minEnclosingTriangle(cnt)

    # Find the three points closest to the triangle corners
    idx, arr = closest_point(tri[0][0],cnt[:,0,:])
    bp1 = cnt[idx, 0, :]
    idx, arr = closest_point(tri[1][0],cnt[:,0,:])
    bp2 = cnt[idx, 0, :]
    idx, arr = closest_point(tri[2][0],cnt[:,0,:])
    bp3 = cnt[idx, 0, :]

    # one of these points is the apex and the other two indicate the MV annulus 
    apex, mv_1, mv_2 = label_points(bp1, bp2, bp3)

    # Find the slope of the line connecting the MV points
    mv_slope = mv_1 - mv_2 

    # PERPINDICULAR BISECTOR
    mid_point = ((mv_1 + mv_2) /2).astype(int)

    # The apex slope should be perpindicular to the mv_slope
    apex_slope = apex - mid_point
    # Normalize the apex slope vector (if the norm exists)
    if np.linalg.norm(apex_slope):
        apex_slope = apex_slope / np.linalg.norm(apex_slope)

    blank = np.zeros(np.shape(frame))
    apex_line = cv2.line(blank.copy(), (apex - 112 * apex_slope).astype(int), (mid_point + 112 * apex_slope).astype(int), 1, 1)
    contour_drawing = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
    intersections = np.logical_and(contour_drawing, apex_line)
    points = np.where(intersections == 1)
    points = np.transpose(points[0:2]) # put points array in the form [[x1, y1], [x2, y2] ...]

    # error handling (ensure there are at least 2 intercept pts, which are far from apex)
    width = 2
    dist = 0

    while np.shape(points)[0] < 2 or dist < 0.90 * (np.linalg.norm(mv_1 - mv_2)):
        apex_line = cv2.line(apex_line, (apex + 10 * apex_slope).astype(int), (mid_point - 10 * apex_slope).astype(int), 1, width)
        contour_drawing = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
        intersections = np.logical_and(contour_drawing, apex_line)
        points = np.where(intersections == 1)
        points = np.transpose(points[0:2])
        idx,arr = farthest_point(np.flip(apex), points)
        dist = arr[idx]

        width = width + 1

    # Choose the point that is furthest away from the apex as the intercept.  
    intercept = points[idx]
    intercept = np.flip(intercept)

    LV_length = np.linalg.norm(apex - intercept)

    V = (8 * (mask_area**2)) / (3 * np.pi * LV_length)
    
    if draw:
        
        # tri = tri.astype(int)
        # p1 = (tri[0][0][0], tri[0][0][1])
        # p2 = (tri[1][0][0], tri[1][0][1])
        # p3 = (tri[2][0][0], tri[2][0][1])
 
        # im3 = cv2.line(im, p1, p3, (0, 255,0), 1)
        # im3 = cv2.line(im3, p3, p2, (0, 255,0), 1)
        # im3 = cv2.line(im3, p1, p2, (0, 255,0), 1)

        #im3 = cv2.drawContours(im, [cnt], 0, (255,255,0), 1)

        # im3 = cv2.circle(im3, apex, 2, (0, 0, 255), 2)
        # im3 = cv2.circle(im3, mv_1, 2, (255, 0, 0), 2)
        # im3 = cv2.circle(im3, mv_2, 2, (255, 0, 0), 2)
        # im3 = cv2.circle(im3, intercept, 2, (0,255,255), 2)

        #im3 = cv2.line(im3, apex, intercept, (255, 0,255), 1)
        overlay = im.copy()
        cv2.fillPoly(overlay, pts =[cnt], color=(253, 231, 37))
        alpha = 0.4
        image_new = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
        im3 = image_new

    return V, im3

def modified_simpsons(cnt, frame, draw, im):
    # Find the minumum enclosing triangle for the LV contour
    [area, tri] = cv2.minEnclosingTriangle(cnt)

    # Find the three points closest to the triangle corners
    idx, arr = closest_point(tri[0][0],cnt[:,0,:])
    bp1 = cnt[idx, 0, :]
    idx, arr = closest_point(tri[1][0],cnt[:,0,:])
    bp2 = cnt[idx, 0, :]
    idx, arr = closest_point(tri[2][0],cnt[:,0,:])
    bp3 = cnt[idx, 0, :]

    # one of these points is the apex and the other two indicate the MV annulus 
    apex, mv_1, mv_2 = label_points(bp1, bp2, bp3)

    mid_point = ((mv_1 + mv_2) /2).astype(int)
    apex_slope = apex - mid_point

    # Normalize the apex slope vector (if the norm exists)
    if np.linalg.norm(apex_slope):
        apex_slope = apex_slope / np.linalg.norm(apex_slope)

    blank = np.zeros(np.shape(frame))
    apex_line = cv2.line(blank.copy(), (apex - 112 * apex_slope).astype(int), (mid_point + 112 * apex_slope).astype(int), 1, 1)
    contour_drawing = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
    intersections = np.logical_and(contour_drawing, apex_line)
    points = np.where(intersections == 1)
    points = np.transpose(points[0:2]) # put points array in the form [[x1, y1], [x2, y2] ...]

    # error handling (ensure there are at least 2 intercept pts, which are far from apex)
    width = 2
    dist = 0

    while np.shape(points)[0] < 2 or dist < 0.90 * (np.linalg.norm(mv_1 - mv_2)):
        apex_line = cv2.line(apex_line, (apex + 10 * apex_slope).astype(int), (mid_point - 10 * apex_slope).astype(int), 1, width)
        contour_drawing = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
        intersections = np.logical_and(contour_drawing, apex_line)
        points = np.where(intersections == 1)
        points = np.transpose(points[0:2])
        idx,arr = farthest_point(np.flip(apex), points)
        dist = arr[idx]

        width = width + 1

    # Choose the point that is furthest away from the apex as the intercept.  
    intercept = points[idx]
    intercept = np.flip(intercept)

    # find 20 points evenly spaced along the biplane 
    x = np.linspace(apex[1], intercept[1], 20)
    y = np.linspace(apex[0], intercept[0], 20)


    pts = np.vstack((y, x)).T

    # slope of connecting line should be perp to the biplane 
    disc_slope = np.flip(apex_slope)
    disc_slope[0] = -disc_slope[0]

    im3 = np.zeros((np.shape(frame)[0], np.shape(frame)[1], 3))

    V = 0
    # for each point compute the volume of the disc 
    for num in range(1, np.shape(pts)[0]-1):
        lower = pts[num]
        upper = pts[num+1]

        height = np.linalg.norm(lower - upper)

        
        i1 = line_contour_intercept(lower, disc_slope, frame, cnt)
        i2 = line_contour_intercept(lower, -disc_slope, frame, cnt)

        #im3 = cv2.line(im3, i1.astype(int), i2.astype(int), (255, 0, 150), 1)

        r = np.linalg.norm(i1 - i2) / 2
        disc_vol = np.pi * (r**2) *height
        V = V + disc_vol

    # print(apex)
    # print(intercept)
    # print(x)
    # print(y)

    if draw:
   

        im3 = cv2.drawContours(im, [cnt], 0, (255,255,0), 1)

        im3 = cv2.circle(im3, apex, 2, (0, 0, 255), 2)
        im3 = cv2.circle(im3, mv_1, 2, (255, 0, 0), 2)
        im3 = cv2.circle(im3, mv_2, 2, (255, 0, 0), 2)
        im3 = cv2.circle(im3, intercept, 2, (0,0,255), 2)

        im3 = cv2.line(im3, apex, intercept, (255, 0,255), 1)

    # for pt in pts:
    #    im3 = cv2.circle(im3, pt.astype(int), 0, (255, 0, 150), -1)

    # filename = 'gif_images/f' + str(index) +'_' + str(i)+ '.png'
    # cv2.imwrite(filename, im3)

    return V, im3

def line_contour_intercept(given_point, slope, frame, cnt):

    blank = np.zeros(np.shape(frame))
    line = cv2.line(blank.copy(), (given_point).astype(int), (given_point + 112 * slope).astype(int), 1, 1)
    contour_drawing = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
    intersections = np.logical_and(contour_drawing, line)
    points = np.where(intersections == 1)
    points = np.transpose(points[0:2])

    width = 2
    while np.shape(points)[0] < 2:
        line = cv2.line(blank.copy(), (given_point).astype(int), (given_point + 112 * slope).astype(int), 1, width)
        contour_drawing = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
        intersections = np.logical_and(contour_drawing, line)
        points = np.where(intersections == 1)
        points = np.transpose(points[0:2])
        width = width + 1

    idx,arr = farthest_point(given_point, points)
    intersection = points[idx]
    intersection = np.flip(intersection)

    return(intersection)

def dice_coef(y_true, y_pred, smooth=0.00001):
    #y_pred = tf.keras.layers.ThresholdedReLU(0.5)(y_pred)
    # y_pred = tf.cast(y_pred, tf.float32)
    # y_true = tf.cast(y_true, tf.float32)
    intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.backend.sum(y_true, axis=[1,2,3]) +keras.backend.sum(y_pred, axis=[1,2,3])
    dice = keras.backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


# echo_train = EchoSet(split = 'train' , pad=8, batch_size = 1)
# echo_valid = EchoSet(split ='val', pad = 8, batch_size = 1)
echo_test = EchoSet(split ='test', pad = 8, batch_size = 1, center = False)

model = RMUNet()
model.load_weights("saved-models/RMU")
model.compile(optimizer=tf.keras.optimizers.Adam())

save_item = True

# # Per frame U-Net
# model = RMUNet()
# model.load_weights("RMU_3channel")

MAE = []
DSC = []
ef_array = []
for item_idx in range(0, echo_test.__len__()):

    per_complete = item_idx * 100 / echo_test.__len__()
   #os.system('clear')
    print( "==========      " , item_idx, "/",echo_test.__len__(), " - ",per_complete, " % complete      ==========")
    print("         Current MAE = ", np.mean(MAE))
    print("         Current DSC = ", np.mean(DSC))
        
    #save_item = True

    X, y = echo_test.__getitem__(item_idx)

    #For per-frame u-net
    outs = np.zeros([1, np.shape(X)[1], 128, 128, 1])
    for m in range(0, np.shape(X)[1] -1):
        out = model.call(X[:, m, :, :, :])
        outs[0][m] = out

    seg_pred = tf.boolean_mask(outs[0], y.ed_es_frames)
    seg_ground_truth =  tf.boolean_mask(y.video_masks, y.ed_es_frames)
    seg_pred = tf.experimental.numpy.where(seg_pred > 0.45, 1.0, 0.0)
    dice = dice_coef(tf.cast(seg_ground_truth, tf.float32), tf.cast(seg_pred, tf.float32))
    DSC = np.append(DSC,dice)
    

    # Calculate the area of each mask in pixels
    seg_pred_avg = np.where(outs > 0.45, 1, 0)
    areas = tf.math.reduce_sum(outs, axis =(2, 3))
    seg_pred_avg = np.squeeze(seg_pred_avg, 0)
    areas = areas[0].numpy()

    #Ensure that frames with no masks are not used for calculation (use arbitrary value of area > 100)
    seg_pred_avg = seg_pred_avg[(areas >100)[:, 0]]
    X = X[:,(areas > 100)[:,0]]
    areas = areas[areas > 100]

    i = 0
    gif_names = []
    vols = np.zeros(np.shape(areas)[0])
    #Skip frames that are zero 
    
    for frame in seg_pred_avg:
        
        frame = frame.astype(np.uint8)
        # Get the contour of the LV
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Find the largest contour
        if contours:

            cnt = contours[0]
            for c in contours:
                area = cv2.contourArea(c)
                if area > cv2.contourArea(cnt):
                    cnt = c
            
            im3 = X[0][i].copy() * 255
            #V, im = modified_simpsons(cnt, frame, draw=True, im=im3)
            V, im = simpsons_biplane(cnt, frame, areas[i],draw=True, im=im3)
            vols[i] = V


            if save_item:
                im3 = im
                filename = 'gif_images/f' + str(i) + '.png'
                gif_names.append(filename)
                cv2.imwrite(filename, im)

        i = i + 1

    if np.any(vols):
        # Reject outliers
        filtered = sig.medfilt(vols)

        # Detect peaks
        trim_min = sorted(filtered)[round(len(filtered) ** 0.05)]
        trim_max = sorted(filtered)[round(len(filtered) ** 0.95)]

        trim_range = trim_max - trim_min
        
        # Distance = minimum required distance between frames, 32 is heuristic
        troughs = sig.find_peaks(-filtered,  distance=20,  prominence=(0.50 * trim_range))

        trough_frames = troughs[0]
        troughs_vols = filtered[trough_frames]
        
        # Uncomment to produce beat level figures 
        # fig, ax = plt.subplots()
        # ax.scatter(range(0, np.shape(filtered)[0]), filtered, color='blue')
        # ylim = ax.get_ylim()

        color = [0.2, 0.9, 0.9]

        beats = []

        for index in range(0, len(trough_frames) -1):
            color_rgb = clr.hsv_to_rgb(color)
            curr_trough = trough_frames[index]
            next_trough = trough_frames[index + 1]

            beat_start = int(curr_trough)
            beat_end = int(next_trough -1)

            #ax.fill_betweenx(ylim, beat_start, beat_end, color= color_rgb, alpha=0.5)

            color[0] = color[0] + 0.2
            beats = np.append(beats, [beat_start, beat_end])
            beats = beats.reshape((-1, 2))


        #plt.show()
        EF_per_beat = []
        for beat in beats:
            curr_beat_vols = filtered[int(beat[0]):int(beat[1])]
            max_frame = np.argmax(curr_beat_vols)
            min_frame = np.argmin(curr_beat_vols)

            EF = (curr_beat_vols[max_frame] - curr_beat_vols[min_frame]) / curr_beat_vols[max_frame]

            #print("Ejection fraction of beat ", b_no, " = ", EF)
            EF_per_beat = np.append(EF_per_beat, EF)

        EF = np.mean(EF_per_beat)
        abs_error = abs(EF -  float(y.ef))
        ef_array = np.append(ef_array, EF)
        MAE = np.append(MAE, abs_error)
        print(EF)
        if np.isnan(MAE).any():
            nans = np.append(nans, item_idx)

    if save_item:
        with imageio.get_writer('gifs/' +str(item_idx) +'.gif', mode='I') as writer:
            for filename in gif_names:
                image = imageio.imread(filename)
                writer.append_data(image)

    for filename in set(gif_names):
        os.remove(filename)

    save_item = True
    MAE = MAE[~np.isnan(MAE)]



#print("MAE: ", MAE / echo_test.__len__())








