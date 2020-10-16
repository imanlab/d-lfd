import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.layers.normalization import BatchNormalization
np.random.seed(1000)

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing

from skimage.io import imread
from skimage.transform import resize




with tf.device('/gpu:0'):  

	################################## Load Robot data ##################################################################
	Arm2_CS_State = pd.read_csv('/home/kiyanoushs/KiyanoushCodes/NeedleInsertion/Data/Arm2_CS_new.csv', header=None)
	Arm2_NS_State = pd.read_csv('/home/kiyanoushs/KiyanoushCodes/NeedleInsertion/Data/Arm2_NS_new.csv', header=None)

	robot_state_train_input = Arm2_CS_State[0:50244][:]
	print("Robot state input trainingset size: {}".format(robot_state_train_input.shape))
	robot_state_train_label = Arm2_NS_State[0:50244][:]
	print("Robot state label trainingset size: {}".format(robot_state_train_label.shape))

	robot_state_test_input = Arm2_CS_State[50244:][:]
	print("Robot state input testset size: {}".format(robot_state_test_input.shape))
	robot_state_test_label = Arm2_NS_State[50244:][:]
	print("Robot state label testset size: {}".format(robot_state_test_label.shape))

	################################## Standardization ###################################################################
	CS_train_names = robot_state_train_input.columns
	NS_train_names = robot_state_train_label.columns

	CS_test_names = robot_state_test_input.columns
	NS_test_names = robot_state_test_label.columns

	scaler = preprocessing.StandardScaler()
	input_Scaler = scaler.fit(robot_state_train_input)
	output_Scaler = scaler.fit(robot_state_train_label)
	robot_state_train_input = input_Scaler.transform(robot_state_train_input)
	robot_state_train_label = output_Scaler.transform(robot_state_train_label)

	robot_state_test_input = input_Scaler.transform(robot_state_test_input)
	robot_state_test_label = output_Scaler.transform(robot_state_test_label)

	robot_state_train_input = pd.DataFrame(robot_state_train_input, columns=CS_train_names)
	robot_state_train_label = pd.DataFrame(robot_state_train_label, columns=NS_train_names)

	robot_state_test_input = pd.DataFrame(robot_state_test_input, columns=CS_test_names)
	robot_state_test_label = pd.DataFrame(robot_state_test_label, columns=NS_test_names)

	robot_state_train_input = np.array(robot_state_train_input)
	robot_state_train_label = np.array(robot_state_train_label)

	robot_state_test_input = np.array(robot_state_test_input)
	robot_state_test_label = np.array(robot_state_test_label)

	############################################### Load image data #####################################################
	X_train_filenames = pd.read_csv('/home/kiyanoushs/KiyanoushCodes/NeedleInsertion/Data/trainImageName.csv', header=None)
	X_test_filenames = pd.read_csv('/home/kiyanoushs/KiyanoushCodes/NeedleInsertion/Data/testImageName.csv', header=None)
	X_train_filenames = np.array(X_train_filenames)
	X_test_filenames = np.array(X_test_filenames)

	X_train_filenames = X_train_filenames[:, 0]
	X_test_filenames = X_test_filenames[:, 0]
	######################################################################################################################

	class My_Custom_Generator(keras.utils.Sequence) :
		
		def __init__(self, image_filenames, robot_input,labels, batch_size) :
			self.image_filenames = image_filenames
			self.robot_input = robot_input
			self.labels = labels
			self.batch_size = batch_size

		def __len__(self) :
			return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

		def __getitem__(self, idx) :
			batch_x_img = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
			batch_x_robot = self.robot_input[idx * self.batch_size : (idx+1) * self.batch_size]
			batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

			return [np.array([
			  	  resize(imread('/home/kiyanoushs/KiyanoushCodes/NeedleInsertion/Data/all_images/' + str(file_name)), (224, 224, 3))
					 	  for file_name in batch_x_img])/255.0, np.array(batch_x_robot)], np.array(batch_y)


	batch_size = 32

	my_training_batch_generator = My_Custom_Generator(X_train_filenames, robot_state_train_input, robot_state_train_label, batch_size)
	my_testing_batch_generator = My_Custom_Generator(X_test_filenames, robot_state_test_input, robot_state_test_label, batch_size)

	########################################################## Define AlexNet CNN ####################################################

	image_input_layer = keras.layers.Input(shape=(224,224,3))

	layer_conv_1 = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation="relu")(image_input_layer)
	layer_pooling_1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_1)

	layer_conv_2 = keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', activation="relu")(layer_pooling_1)
	layer_pooling_2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_2)

	layer_conv_3 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_pooling_2)
	layer_conv_4 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_conv_3)
	layer_conv_5 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_conv_4)

	layer_pooling_3 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_5)

	cnn_flatten = keras.layers.Flatten()(layer_pooling_3)

	dense_1 = keras.layers.Dense(4096, activation="relu")(cnn_flatten)
	drop_1 = keras.layers.Dropout(0.4)(dense_1)
	dense_2 = keras.layers.Dense(4096, activation="relu")(drop_1)
	drop_2 = keras.layers.Dropout(0.4)(dense_2) 


	robot_state_input_layer = keras.layers.Input(shape=(7,))

	dense_3 = keras.layers.Dense(15, activation="relu")(robot_state_input_layer)
	dense_4 = keras.layers.Dense(25, activation="relu")(dense_3)

	concat = keras.layers.concatenate([dense_4 , drop_2])

	dense_5 = keras.layers.Dense(80, activation="relu")(concat)
	dense_6 = keras.layers.Dense(20, activation="relu")(dense_5)
	output_layer = keras.layers.Dense(7, activation="linear")(dense_6)

	model = keras.models.Model(inputs=[image_input_layer , robot_state_input_layer] , outputs=output_layer)

	# Compile the model
	model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mse','accuracy'])

	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, 
	        verbose=1, mode='auto', restore_best_weights=True)

	history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch =int(50244 // batch_size), callbacks=[monitor], epochs=7)
	#score = model.evaluate([test , robot_state_test_input] , robot_state_test_label) 


	#predict_AlexNet_dense = model.predict([test , robot_state_test_input])

	##################################### save Model ###########################################################################
	model.save('AlexNet.h5')

	#model.summary()