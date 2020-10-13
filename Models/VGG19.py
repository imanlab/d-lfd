import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



with tf.device('/gpu:1'):

	Arm1_CS_State = pd.read_csv('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Arm1_CS.csv')
	Arm1_NS_State = pd.read_csv('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Arm1_NS.csv')

	#X = np.load('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Test1-20imageDataColor.npy')
	X = np.load('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Test1-20resizedimageDataColor.npy')

	print(Arm1_CS_State[0:1])

	D3data = np.ones((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
	D3data[ : , : , : , :] = X
	print(D3data.shape)


	train = D3data[0:D3data.shape[0]-2000 , : , : , :]
	train = train.astype(np.float64)
	print("training dataset size : {}".format(train.shape[0]))

	for i in range(0 , train.shape[0]):
	  train[i, : , : , :] = train[i, : , : , :] / np.max(train[i, : , : , :])

	robot_state_train_input = Arm1_CS_State[0:train.shape[0]]
	print("Robot state input trainingset size: {}".format(robot_state_train_input.shape))
	robot_state_train_label = Arm1_NS_State[0:train.shape[0]]
	print("Robot state label trainingset size: {}".format(robot_state_train_label.shape))

	test = D3data[D3data.shape[0]-2000: , : , : , :]
	test = test.astype(np.float64)

	for i in range(0 , test.shape[0]):
	  test[i, : , : , :] = test[i, : , : , :] / np.max(test[i, : , : , :])

	robot_state_test_input = Arm1_CS_State[train.shape[0]:train.shape[0]+test.shape[0]]
	print("Robot state input testset size: {}".format(robot_state_test_input.shape))
	robot_state_test_label = Arm1_NS_State[train.shape[0]:train.shape[0]+test.shape[0]]
	print("Robot state label testset size: {}".format(robot_state_test_label.shape))


	model = VGG19(include_top=False, weights='imagenet', input_shape=(224 , 224 , 3))
	#model.summary()


	for layer in model.layers[:21]:
	    layer.trainable=False
	for layer in model.layers[21:]:
	    layer.trainable=True


	y1 = model.output
	y2 = GlobalAveragePooling2D()(y1)
	y3 = Dense(512,activation='relu')(y2) 
	y4 = Dense(512,activation='relu')(y3) 
	#preds = Dense(7,activation='linear')(y)

	new_model = Model(inputs=model.input,outputs=y4)


####################################################################################################################
	cnn_out = new_model.output


	robot_state_input_layer = keras.layers.Input(shape=(7,))

	dense_1 = keras.layers.Dense(15, activation="relu")(robot_state_input_layer)
	dense_2 = keras.layers.Dense(25, activation="relu")(dense_1)

	concat = keras.layers.concatenate([dense_2 , cnn_out])

	dense_3 = keras.layers.Dense(80, activation="relu")(concat)
	dense_4 = keras.layers.Dense(20, activation="relu")(dense_3)
	output_layer = keras.layers.Dense(7, activation="linear")(dense_4)

	vgg19_model = keras.models.Model(inputs=[new_model.input , robot_state_input_layer] , outputs=output_layer)

#####################################################################################################################




	vgg19_model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mse','accuracy'])

	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, 
	        verbose=1, mode='auto', restore_best_weights=True)

	history = vgg19_model.fit([train , robot_state_train_input], robot_state_train_label, callbacks=[monitor], batch_size=20, validation_split=0.2, epochs=6)
	score = vgg19_model.evaluate([test , robot_state_test_input] , robot_state_test_label)


	predict_vgg_dense = vgg19_model.predict([test , robot_state_test_input])


	err_matrix_vgg_dense = robot_state_test_label - predict_vgg_dense
	vgg_err_mean = np.mean(abs(err_matrix_vgg_dense))
	print("vgg19 mean error values for each output: ")
	print(vgg_err_mean)
	a = np.where(err_matrix_vgg_dense > 0.01)
	a = np.asarray(list(zip(*a)))
	print("number of err elements higher than 0.01: {}".format(a.shape))


	print(predict_vgg_dense.shape)
	vgg19_model.summary()


	vgg19_model.save('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/VGG19.h5')