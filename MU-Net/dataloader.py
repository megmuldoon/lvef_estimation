import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import collections
import cv2
import skimage.draw
import matplotlib.pyplot as plt

print(tf.__version__)

class EchoMasks(tf.keras.utils.Sequence):
	def __init__(self, root = "EchoNet-Dynamic/",split='train', batch_size = 1, noise = None, padding = None):
		self.folder = root
		self.split = split
		self.batch_size = batch_size
		self.noise = noise
		self.pad = padding

		self.fnames =[]
		self.outcomes =[]
		self.ejection=[]
		self.fps = []

		with open(self.folder + "FileList.csv") as f:
			header   = f.readline().strip().split(",")
			filenameIndex = header.index("FileName")
			splitIndex = header.index("Split")
			efIndex = header.index("EF")
			fpsIndex = header.index("FPS")
			for line in f:
				lineSplit = line.strip().split(',')
				# Get name of the video file
				fileName = os.path.splitext(lineSplit[filenameIndex])[0]+".avi"
				
				#Get the subset that the video belongs to 
				fileSet = lineSplit[splitIndex].lower()
				
				#Get ef for the video
				fileEf = lineSplit[efIndex]

				#Get fps for the video
				fileFps = lineSplit[fpsIndex]

				#Ensure that the video exists 
				if os.path.exists(self.folder + "/Videos/" + fileName):
					if fileSet == split:
						self.fnames.append(fileName)
						self.outcomes.append(lineSplit)
						self.ejection.append(fileEf)
						self.fps.append(fileFps)

		self.frames = collections.defaultdict(list)
		_defaultdict_of_lists_ = collections.defaultdict(list)
		self.trace = collections.defaultdict(lambda: collections.defaultdict(list))

		#Read the voilume tracings CSV file to find videos with ED/ES frames 
		with open(self.folder + "VolumeTracings.csv") as f:
			header = f.readline().strip().split(",")
			assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

			for line in f:
				filename, x1, y1, x2, y2, frame = line.strip().split(",")
				x1 = float(x1)
				x2 = float(x2)
				y1 = float(y1)
				y2 = float(y2)
				frame = int(frame)

				# New frame index for the given filename
				if frame not in self.trace[filename]:
					self.frames[filename].append(frame)
				self.trace[filename][frame].append((x1,y1,x2,y2))

		#Transform into numpy array 
		for filename in self.frames:
			for frame in self.frames[filename]:
				self.trace[filename][frame] = np.array(self.trace[filename][frame])

		# Reject files without tracings 
		keep = [len(self.frames[f]) >= 2 for f in self.fnames]
		self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
		self.outcomes = [f for (f, k) in zip(self.outcomes, keep) if k]

		self.indexes = np.arange(np.shape(self.fnames)[0])


	def __len__(self):
		# Denotes the number of batches per epoch
		return(int (np.floor(np.shape(self.fnames)[0])/self.batch_size))

	def __getitem__(self, idx):
		# Generate one batch of data
		# Generate indexes of the batch

		indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
		X, y = self.__data_generation(indexes)


		return X, y


	def __data_generation(self, list_IDs_temp):

		X = []
		y = []

		index = 0
		for i in list_IDs_temp:
			path = os.path.join(self.folder, "Videos", self.fnames[i])
			# Load video into np.array
			if not os.path.exists(path):
				print("File does not exist")

			frames = self.frames[self.fnames[i]]
			frames.sort() #  Ensure that the frames are in ascending order
			traces = self.trace[self.fnames[i]]

			vid_cap = cv2.VideoCapture(path)
			frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
			frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

			# Read the entire video and save the traced frames 
			inputs = np.zeros((len(frames),frame_width, frame_height, 3), np.uint8)
			targets = np.zeros((len(frames),frame_width, frame_height), np.uint8)
			index = 0
			
			# Load the frames  
			for count in range(frame_count): 
				success, frame = vid_cap.read()
				if not success:
					print("Failed to load frame #", count, ' of ', filename)
				
				if (count) in frames: #Assume that frame number is 1 indexed
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					inputs[index] = frame
					index = index + 1
			
			# blackout pixels for simulated noise
			if self.noise:
				num_pepper = np.ceil(self.noise * 2 * frame_height * frame_width)
				coords = [np.random.randint(0, i-1, int(num_pepper)) for i in inputs.shape[0:3]]
				inputs[tuple(coords)] = (0,0,0)
			
			# Scale pixels between 0 and 1
			inputs = inputs /255.0
			X = np.append(X, inputs)
			X = X.reshape((-1, 112,112,3))
	
			#Load the targets
			index = 0
			for f in frames:
				t = traces[f]
				x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
				x = np.concatenate((x1[1:], np.flip(x2[1:])))
				y_ = np.concatenate((y1[1:], np.flip(y2[1:])))
				r, c = skimage.draw.polygon(np.rint(y_).astype(np.int), np.rint(x).astype(np.int), (frame_width,frame_height))
				mask = np.zeros((frame_width, frame_height), np.float32)
				mask[r, c] = 1
				targets[index] = mask
				index = index + 1	

			y = np.append(y, targets)
			y = y.reshape(-1, 112, 112)

		y = np.expand_dims(y, -1)
		
		if self.pad is not None:
			X = np.pad(X, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), mode='constant', constant_values=0)
			y = np.pad(y, ((0,0),(self.pad,self.pad),(self.pad,self.pad), (0,0)), mode='constant', constant_values=0)

		return X, y

	def display_example(self, idx):
		indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
		X, y = self.__data_generation(indexes)

		fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
		axs[0].imshow(X[0])
		axs[0].set_title("Frame")
		axs[1].imshow(y[0])
		axs[1].set_title("Mask")
		plt.show()


# Define packed tensor so that we can have a segmentation mask ground truth and a EF ground truth
class PackedTensor(tf.experimental.BatchableExtensionType):
    __name__ = 'extension_type_colab.PackedTensor'

    ef: tf.Tensor
    video_masks: tf.Tensor
    ed_es_frames: tf.Tensor
    edv : tf.Tensor
    esv : tf.Tensor

    # shape and dtype hold no meaning in this context, so we use a dummy
    # to stop Keras from complaining

    shape = property(lambda self: self.output_0.shape)
    dtype = property(lambda self: self.output_0.dtype)

    class Spec:

        def __init__(self, shape, dtype=tf.float32):

        	self.ef = tf.TensorSpec(shape, dtype)
        	self.video_masks = tf.TensorSpec(shape, dtype)
        	self.ed_es_frames = tf.TensorSpec(shape, dtype)
        	self.edv = tf.TensorSpec(shape, dtype)
        	self.esv = tf.TensorSpec(shape, dtype)

        # shape and dtype hold no meaning in this context, so we use a dummy
        # to stop Keras from complaining
        shape: tf.TensorShape = tf.constant(1.).shape 
        dtype: tf.DType = tf.constant(1.).dtype

# these two functions have no meaning, but need dummy implementations
# to stop Keras from complaining
@tf.experimental.dispatch_for_api(tf.shape)
def packed_shape(input: PackedTensor, out_type=tf.int32, name=None):
    return tf.shape(input.col_ids)

@tf.experimental.dispatch_for_api(tf.cast)
def packed_cast(x: PackedTensor, dtype: str, name=None):
    return x

class EchoSet(tf.keras.utils.Sequence):
	def __init__(self, root = "EchoNet-Dynamic/", split='train', max_length = 128, min_spacing = 16, pad = 8, attenuation = 3, batch_size =1,  center=True):
		self.folder = root
		self.split = split
		self.max_length = max_length
		self.min_length = min_spacing
		self.pad = pad
		self.attenuation = attenuation
		self.batch_size = batch_size
		self.center = center


		self.fnames =[]
		self.outcomes =[]
		self.ejection=[]
		self.fps = []
		self.edv = []
		self.esv = []
		
		with open(self.folder + "FileList.csv") as f:
			header   = f.readline().strip().split(",")
			filenameIndex = header.index("FileName")
			splitIndex = header.index("Split")
			efIndex = header.index("EF")
			fpsIndex = header.index("FPS")

			edvIndex = header.index("EDV")	
			esvIndex = header.index("ESV")


			for line in f:
				lineSplit = line.strip().split(',')
				# Get name of the video file
				fileName = os.path.splitext(lineSplit[filenameIndex])[0]+".avi"
				
				#Get the subset that the video belongs to 
				fileSet = lineSplit[splitIndex].lower()
				
				#Get ef for the video
				fileEf = lineSplit[efIndex]

				#Get fps for the video
				fileFps = lineSplit[fpsIndex]

				# Get the EDV and ESV for the video 
				fileEDV = lineSplit[edvIndex]
				fileESV = lineSplit[esvIndex]

				#Ensure that the video exists 
				if os.path.exists(self.folder + "/Videos/" + fileName):
					if fileSet == split:
						self.fnames.append(fileName)
						self.outcomes.append(lineSplit)
						self.ejection.append(fileEf)
						self.fps.append(fileFps)
						self.edv.append(fileEDV)
						self.esv.append(fileESV)

		self.frames = collections.defaultdict(list)
		_defaultdict_of_lists_ = collections.defaultdict(list)
		self.trace = collections.defaultdict(lambda: collections.defaultdict(list))

		#Read the voilume tracings CSV file to find videos with ED/ES frames 
		with open(self.folder + "VolumeTracings.csv") as f:
			header = f.readline().strip().split(",")
			assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

			for line in f:
				filename, x1, y1, x2, y2, frame = line.strip().split(",")
				x1 = float(x1)
				x2 = float(x2)
				y1 = float(y1)
				y2 = float(y2)
				frame = int(frame)

				# New frame index for the given filename
				if frame not in self.trace[filename]:
					self.frames[filename].append(frame)
				self.trace[filename][frame].append((x1,y1,x2,y2))

		#Transform into numpy array 
		for filename in self.frames:
			for frame in self.frames[filename]:
				self.trace[filename][frame] = np.array(self.trace[filename][frame])

		# Reject files without tracings 
		keep = [len(self.frames[f]) >= 2 for f in self.fnames]
		self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
		self.outcomes = [f for (f, k) in zip(self.outcomes, keep) if k]
		self.indexes = np.arange(np.shape(self.fnames)[0])

		self.indexes = np.arange(np.shape(self.fnames)[0])

	def __len__(self):
		# Denotes the number of batches per epoch
		return(int (np.floor(np.shape(self.fnames)[0])/self.batch_size))

	def __getitem__(self, idx):
		# Generate one batch of data and generate indexes of the batch
		indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
		X, ef, masks, ed_es_frames, edv, esv = self.__data_generation(indexes)
		X = np.expand_dims(X, 0)


		ed_es_frames = np.array(ed_es_frames, dtype =bool)

		# use packed tensor so that label contains the ef and the masks
		packed_output = PackedTensor(ef, masks, ed_es_frames, edv, esv)

		return X, packed_output
	
	def __data_generation(self, list_IDs_temp):
		# Retun the video (starting at the first annotated frame) and the mask of the first annotation 
		#X = np.empty((self.batch_size, self.max_length, 3, 128,128))
		X= []
		ejection = np.empty(self.batch_size)
		edv = np.empty(self.batch_size)
		esv = np.empty(self.batch_size)

		index = 0
		for i in list_IDs_temp:
			path = os.path.join(self.folder, "Videos", self.fnames[i])
			# Load video into np.array
			if not os.path.exists(path):
				print("File does not exist")

			vid_cap = cv2.VideoCapture(path)
			f = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
			w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

			v = np.zeros((f, w, h, 3), np.uint8)

			for count in range(f):
				success, frame = vid_cap.read()
				if not success:
					print("Failed to load frame #", count, ' of ', filename)
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				v[count] = frame

			video = v.transpose((3,0,1,2))
			key = os.path.splitext(self.fnames[i])[0]

			video = video / 255.0
			video = np.moveaxis(video,0,1)

			frames = self.frames[self.fnames[i]]
			frames.sort()
			traces = self.trace[self.fnames[i]]
			first_frame = frames[0]

			nvideo = []

			# Start from frame ED -1 (if centering)
			# In some cases the test videos have fewer than max_len frames
			if (self.center):
				if  f - first_frame + 1 > self.max_length:
					nvideo.append(video[first_frame -1])
					nvideo.extend(video[first_frame:first_frame + self.max_length -1])
				else:
					nvideo.append(video[first_frame -1 ])
					nvideo.extend(video[first_frame:f])

					# zeros = np.zeros( (3, w, h))
					# for i in range(0, self.max_length - f + first_frame -1):
					# 	nvideo.append(zeros)

			# return all frames
			else:
				nvideo.append(video[0])
				nvideo.extend(video[1:f])


			#Load the mask for ES and ED
			if self.center:
				masks = np.empty((self.batch_size,np.shape(nvideo)[0], 128, 128, 1))
				ed_es_frames = np.zeros((self.batch_size, np.shape(nvideo)[0]))

				for f_no in frames:
					
					t = traces[f_no]
					x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
					x = np.concatenate((x1[1:], np.flip(x2[1:])))
					y_ = np.concatenate((y1[1:], np.flip(y2[1:])))
					r, c = skimage.draw.polygon(np.rint(y_).astype(np.int), np.rint(x).astype(np.int), (w,h))
					mask = np.zeros((w,h), np.float32)
					mask[r, c] = 1
					mask = np.pad(mask, ((self.pad,self.pad),(self.pad,self.pad)), mode='constant', constant_values=0)
					mask = np.reshape(mask, (128,128, 1))

					if(f_no - first_frame < self.max_length):
						masks[index][f_no - first_frame] = mask
						ed_es_frames[index][f_no - first_frame] = 1
			else:
				masks = []
				ed_es_frames = []
				zeros = np.zeros((128, 128,1))

				for f_no in range(0, f):
					# if this frame has a trace
					if f_no in frames:
						t = traces[f_no]
						x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
						x = np.concatenate((x1[1:], np.flip(x2[1:])))
						y_ = np.concatenate((y1[1:], np.flip(y2[1:])))
						r, c = skimage.draw.polygon(np.rint(y_).astype(np.int), np.rint(x).astype(np.int), (w,h))
						mask = np.zeros((w,h), np.float32)
						mask[r, c] = 1
						mask = np.pad(mask, ((self.pad,self.pad),(self.pad,self.pad)), mode='constant', constant_values=0)
						mask = np.reshape(mask, (128,128, 1))
						masks.append(mask)
						ed_es_frames.append(1)

						
					else:
						masks.append(zeros)
						ed_es_frames.append(0)




			# Padding 
			nvideo = np.pad(nvideo, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), mode='constant', constant_values=0)
			nvideo = np.swapaxes(nvideo, 1, 2)
			nvideo = np.swapaxes(nvideo, 2, 3)

			X = np.append(X, nvideo)
			X = X.reshape((-1, 128,128,3))

			
			ejection[index] = float(self.ejection[i]) / 100.0 #Divide by 10 to put in range 0-1
			edv[index] = float(self.edv[i])
			esv[index] = float(self.esv[i])
			index = index + 1

			#print(np.shape(self.edv))
			
			
		return tf.convert_to_tensor(X), tf.convert_to_tensor(ejection), tf.convert_to_tensor(masks), tf.convert_to_tensor(ed_es_frames), tf.convert_to_tensor(edv), tf.convert_to_tensor(esv)
