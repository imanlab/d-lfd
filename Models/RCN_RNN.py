import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model



with tf.device('/gpu:1'):

	Arm1_CS_State = pd.read_csv('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Arm1_CS.csv')
	Arm1_NS_State = pd.read_csv('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Arm1_NS.csv')

	#X = np.load('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Test1-20imageDataGray.npy')
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





	intermediate_layer_model = load_model('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/CNN_intermediate_layer.h5')

	intermediate_output_train = intermediate_layer_model.predict([train, robot_state_train_input])
	intermediate_output_test = intermediate_layer_model.predict([test, robot_state_test_input])


	intermediate_output_train.shape

	validation_set_size = 1500

	train_set = intermediate_output_train[0:intermediate_output_train.shape[0]-validation_set_size , : ]
	validation_set = intermediate_output_train[intermediate_output_train.shape[0]-validation_set_size: , : ]
	test_set = intermediate_output_test[: , :]

	timeWindow = 4

	train = keras.preprocessing.sequence.TimeseriesGenerator(train_set, train_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)
	validation = keras.preprocessing.sequence.TimeseriesGenerator(validation_set, validation_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)
	test = keras.preprocessing.sequence.TimeseriesGenerator(test_set, test_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)

	train[0][0].shape


	print("train_set shape: {}".format(train_set.shape))



	train_matrix = np.zeros((train_set.shape[0]-timeWindow, timeWindow, 80))
	for i in range(timeWindow,train_set.shape[0]):
	  train_matrix[i-timeWindow, : , : ] = train[i-timeWindow][0][0]


	validation_matrix = np.zeros((validation_set.shape[0]-timeWindow, timeWindow, 80))
	for i in range(timeWindow,validation_set.shape[0]):
	  validation_matrix[i-timeWindow, : , : ] = validation[i-timeWindow][0][0]


	test_matrix = np.zeros((test_set.shape[0]-timeWindow, timeWindow, 80))
	for i in range(timeWindow,test_set.shape[0]):
	  test_matrix[i-timeWindow, : , : ] = test[i-timeWindow][0][0]


	output_train_matrix = np.zeros((train_set.shape[0]-timeWindow, 7))
	for i in range(timeWindow,train_set.shape[0]):
	  output_train_matrix[i-timeWindow , :] = robot_state_train_label[i-timeWindow:(i-timeWindow+1)]


	output_validation_matrix = np.zeros((validation_set.shape[0]-timeWindow, 7))
	for i in range(timeWindow,validation_set.shape[0]):
	  output_validation_matrix[i-timeWindow, : ] = robot_state_train_label[(intermediate_output_train.shape[0] - validation_set_size + i - timeWindow):(intermediate_output_train.shape[0] - validation_set_size + i - timeWindow)+1]


	output_test_matrix = np.zeros((test_set.shape[0]-timeWindow, 7))
	for i in range(timeWindow,test_set.shape[0]):
	  output_test_matrix[i - timeWindow,:] = robot_state_test_label[(i - timeWindow):(i - timeWindow)+1]

	print("output_test_matrix: {}".format(output_test_matrix.shape))



	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
	        verbose=1, mode='auto', restore_best_weights=True)


	RNN_model = keras.models.Sequential()
	RNN_model.add(keras.layers.SimpleRNN(150, return_sequences=True, input_shape=(timeWindow,80 )))
	RNN_model.add(keras.layers.SimpleRNN(100, input_shape=(timeWindow,80 )))
	RNN_model.add(keras.layers.Dense(50))
	RNN_model.add(keras.layers.Dense(40))
	RNN_model.add(keras.layers.Dense(7,activation="linear"))
	RNN_model.compile(loss='mae', optimizer=keras.optimizers.Adam(), metrics=['mse','accuracy'])
	RNN_model.fit(train_matrix ,output_train_matrix, callbacks=[monitor], epochs=40, validation_data=(validation_matrix, output_validation_matrix),verbose=2)
	score= RNN_model.evaluate(test_matrix, output_test_matrix) 


	test_predict_RNN = RNN_model.predict(test_matrix)

	err_RNN_Model= test_predict_RNN - output_test_matrix
	RNN_err_mean = np.mean(abs(err_RNN_Model))
	a = np.where(err_RNN_Model > 0.01)
	a = np.asarray(list(zip(*a)))
	print("number of err elements higher than 0.01")
	print(a.shape)

	RNN_model.save('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/RCN_RNN.h5')

	RNN_model.summary()