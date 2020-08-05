#!/usr/bin/python3
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

train_batches=2000
eval_batches=50
train_sequlen=32
train_inputs=1
lstm_states=6
#activation="relu"
activation=None
rec_activation="hard_sigmoid"

x_train = np.zeros((train_batches*train_sequlen,1,train_inputs))
y_train = np.zeros((train_batches*train_sequlen,1,1))
x_test = np.zeros((eval_batches*train_sequlen,1,train_inputs))
y_test = np.zeros((eval_batches*train_sequlen,1,1))

random.seed(1234)

# generate input of random sine waves, feed one at a time to the network

def random_sample():
    ampl = random.uniform(0.5,1)
    freq = random.uniform(18,32)
    phase= random.uniform(-math.pi,math.pi)
    return (ampl,freq,phase)

def waveform(ampl,freq,phase,idx):
    return ampl*math.sin(idx/freq*2*math.pi+phase)

# calculate train data
for i in range(train_batches):
    (ampl,freq,phase) = random_sample()
    for j in range(train_sequlen): # subsequent measurements
        for k in range(train_inputs):
            x_train[i*train_sequlen+j][0][k]=waveform(ampl,freq,phase,j+k)
        y_train[i*train_sequlen+j][0]=waveform(ampl,freq,phase,j+train_inputs)
for i in range(eval_batches):
    (ampl,freq,phase) = random_sample()
    for j in range(train_sequlen): # subsequent measurements
        for k in range(train_inputs):
            x_test[i*train_sequlen+j][0][k]=waveform(ampl,freq,phase,j+k)
        y_test[i*train_sequlen+j][0]=waveform(ampl,freq,phase,j+train_inputs)

print(x_train[0][0:5], y_train[0][0:5])
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

def create_model(train=True):

    if train:
        input0 = tf.keras.Input(batch_shape=(train_sequlen,1,train_inputs))
        # stateful is worse
        x = LSTM(lstm_states, recurrent_activation=rec_activation, activation=activation, return_sequences=False, return_state=False, stateful=False)(input0)
        #x = Dropout(0.1)(x) makes it a bit worse
    else:
        input0 = tf.keras.Input(batch_shape=(1,1,train_inputs),name="data")
        input1 = tf.keras.Input(batch_shape=(1,lstm_states),name="state_h")
        input2 = tf.keras.Input(batch_shape=(1,lstm_states),name="state_c")
        x, state,state2 = LSTM(lstm_states, recurrent_activation=rec_activation, activation=activation, return_sequences=False, return_state=True, stateful=True, unroll=True)(input0, initial_state=(input1, input2))

    x = Dense(units=1)(x)

    if train:
        model = tf.keras.Model(inputs=input0, outputs=x, name="sine")
    else:
        model = tf.keras.Model(inputs=(input0,input1,input2), outputs=(x,state,state2), name="sine")
    model.summary()
    return model

model=create_model()

model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(8):
	model.fit(x_train, y_train, epochs=1, batch_size=train_sequlen, verbose=1, shuffle=False,
        validation_data=(x_test,y_test))
	model.reset_states()

model.save('mymodel')
model.save('mymodel_w.h5', save_format="h5")

model2= create_model(False)
model2.load_weights('mymodel_w.h5')
model2.save('evalmodel.h5', save_format="h5")

model2.compile(loss='mean_squared_error', optimizer='adam')

state_h2 = np.zeros((1,lstm_states))
state_c2 = np.zeros((1,lstm_states))
for i in range(train_sequlen):
	testx, testy = x_test[i], y_test[i]
	testx = testx.reshape(1, 1, 1)
	res = model2.predict([testx,state_h2,state_c2], batch_size=1)
	print('In=%.1f, Expected=%.1f, Predicted=%.1f' % (testx[0][0][0], testy, res[0]))
	state_h2=res[1]
	state_c2=res[2]

# to convert to tflite use
# tflite_convert --keras_model_file evalmodel.h5 --output_file evalmodel.tflite  --inference_type FLOAT
# from tensorflow 1.15 (2.2 doesn't work)
