import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing



with tf.device('/gpu:1'):

	Arm1_CS_State = pd.read_csv('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Arm2_CS_ordered.csv', header=None)
	Arm1_NS_State = pd.read_csv('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Arm2_NS_ordered.csv', header=None)


	CS1_names = Arm1_CS_State.columns
	NS1_names = Arm1_NS_State.columns
	scaler = preprocessing.StandardScaler()
	myScaler = scaler.fit(Arm1_CS_State)
	Arm1_CS_State = myScaler.transform(Arm1_CS_State)
	Arm1_NS_State = myScaler.transform(Arm1_NS_State)

	Arm1_CS_State = pd.DataFrame(Arm1_CS_State, columns=CS1_names)
	Arm1_NS_State = pd.DataFrame(Arm1_NS_State, columns=NS1_names)


	X = np.load('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Test1-20imageOrdered.npy')
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



	#image_input_layer = keras.layers.Input(shape=(174,224,1))
	image_input_layer = keras.layers.Input(shape=(224,224,3))

	layer_conv_1 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(image_input_layer)
	layer_conv_2 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_1)
	layer_pooling_1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_conv_2)

	layer_conv_3 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(layer_pooling_1)
	layer_conv_4 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_3)
	layer_pooling_2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_conv_4)

	layer_conv_5 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(layer_pooling_2)
	layer_conv_6 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_5)
	layer_conv_7 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_6)
	layer_pooling_3 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_conv_7)

	layer_conv_8 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_pooling_3)
	layer_conv_9 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_8)
	layer_conv_10 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_9)
	layer_pooling_4 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_conv_10)

	layer_conv_11 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_pooling_4)
	layer_conv_12 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_11)
	layer_conv_13 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(layer_conv_12)

	cnn_flatten = keras.layers.Flatten()(layer_conv_13)

	robot_state_input_layer = keras.layers.Input(shape=(7,))
	dense_1 = keras.layers.Dense(15, activation="relu")(robot_state_input_layer)
	dense_2 = keras.layers.Dense(25, activation="relu")(dense_1)

	concat = keras.layers.concatenate([dense_2 , cnn_flatten])

	dense_3 = keras.layers.Dense(80, activation="relu")(concat)
	dense_4 = keras.layers.Dense(20, activation="relu")(dense_3)
	output_layer = keras.layers.Dense(7, activation="linear")(dense_4)

	model = keras.models.Model(inputs=[image_input_layer , robot_state_input_layer] , outputs=output_layer)
	model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mse','accuracy'])

	latent_space = 'dense_3'
	intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=dense_3)



	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, 
	        verbose=1, mode='auto', restore_best_weights=True)

	history = model.fit([train , robot_state_train_input], robot_state_train_label, callbacks=[monitor], batch_size=50, validation_split=0.2, epochs=6)
	score = model.evaluate([test , robot_state_test_input] , robot_state_test_label)


	predict_cnn_dense = model.predict([test , robot_state_test_input])


	err_matrix_cnn_dense = robot_state_test_label - predict_cnn_dense
	cnn_err_mean = np.mean(abs(err_matrix_cnn_dense))
	print("CNN mean error values for each output: ")
	print(cnn_err_mean)
	a = np.where(err_matrix_cnn_dense > 0.01)
	a = np.asarray(list(zip(*a)))
	print("number of err elements higher than 0.01: {}".format(a.shape))


	predict_cnn_dense.shape

	model.save('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/CNN_Dense_Net.h5')
	intermediate_layer_model.save('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/CNN_intermediate_layer.h5')

	model.summary()



