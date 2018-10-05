
#%% importing routine libraries

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input,Convolution2D, BatchNormalization, concatenate,AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Reshape
from keras.utils import np_utils
from keras.layers.merge import add
from keras import backend as K
from keras.layers.recurrent import GRU
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from contextlib import redirect_stdout
import model_module as mm
import pickle

#%% loading up the dataset(pictures) and the ground truth (labels)

data_x= np.load('../data/data_aumentation_matrix_representations.npy')
data_y= np.load('../data/ground_truth_data_augmentation.npy')

#%% creation of the labels, we need to create them for both Fonts and Characters because there are strings
    #The bold and the italics are already in binary format

font_names = list(set(data_y[:,0]))
char_names = list(set(data_y[:,1]))

#preparing the one hot encoder to use later
oh = OneHotEncoder()

font_encoding, char_encoding, inverted_font_encoding, inverted_char_encoding= mm.create_labels(data_y)

#save the encodings
pickle.dump(font_encoding, open( "../data/font_encoding.p", "wb" ))
pickle.dump(inverted_font_encoding, open( "../data/inverted_font_encoding.p", "wb" ))
pickle.dump(char_encoding, open( "../data/char_encoding.p", "wb" ))
pickle.dump(inverted_char_encoding, open( "../data/inverted_char_encoding.p", "wb" ))

#%% Data preparation

#split the datasets in 
#X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=.3, train_size = .7)
X_train = data_x
Y_train = data_y

#split the datasets in 
#X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=.3, train_size = .7)

# Loading the dependent data and transforming it into a onehotencoded format,
Y_train = [oh.fit_transform(Y_train[:,0:1]).todense(),
           oh.fit_transform(Y_train[:,1:2]).todense(),
           oh.fit_transform(Y_train[:,2:3]).todense(),
           oh.fit_transform(Y_train[:,3:4]).todense()]

#%% Model Building

#getting the dimentions of the input
input_shape = X_train.shape[1:] 


inp = Input(shape=input_shape)

'''
model3 - .33
conv_1 = Convolution2D(filters = 96, kernel_size=(3, 3), strides = (2,2),  padding = 'same', activation = 'relu') (inp)
pool_1 = MaxPooling2D(pool_size=2, strides =(2,2)) (conv_1)
norm_1 = BatchNormalization() (pool_1)

conv_2 = Convolution2D(filters = 256, kernel_size=(5, 5), padding = 'same', activation = 'relu') (norm_1)
pool_2 = MaxPooling2D(pool_size=3, strides =(2,2)) (conv_2)
norm_2 = BatchNormalization() (pool_2)

conv_3 = Convolution2D(filters = 384, kernel_size=(3, 3), padding = 'same', activation = 'relu') (norm_2)
conv_4 = Convolution2D(filters = 384, kernel_size=(3, 3), padding = 'same', activation = 'relu') (conv_3)
conv_5 = Convolution2D(filters = 256, kernel_size=(3, 3), padding = 'same', activation = 'relu') (conv_4)
pool_3 = MaxPooling2D(pool_size=3, strides =(2,2)) (conv_5)
norm_3 = BatchNormalization() (pool_3)

flat = Flatten()(norm_3)
net_1 = Dense(1080, activation='tanh')(flat) 
drop_1 = Dropout(0.5) (net_1)

net_2 = Dense(1080, activation='tanh')(drop_1) 
drop_2 = Dropout(0.5) (net_2)
'''

#model 2 - .57
conv_1 = Convolution2D(filters = 16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inp)
pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
conv_2 = Convolution2D(filters = 16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(pool_1)
pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
conv_to_rnn_dims = (64 // (2 ** 2), (64 // (2 ** 2)) * 16)
reshape= Reshape(target_shape=conv_to_rnn_dims)(pool_2)
dense_1 = Dense(32, activation='relu', name='dense1')(reshape)
gru_1 = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru1')(dense_1)
gru_1b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(dense_1)
gru1_merged = add([gru_1, gru_1b])
gru_2 = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
gru2_merged = concatenate([gru_2, gru_2b])



dense_2 = Dense(64, kernel_initializer='he_normal', name='dense2')(gru2_merged)
flat = Flatten()(dense_2)


'''
#model 1 - 0.50
strides=(1, 1)
conv_1 = Convolution2D(filters = 32, kernel_size=(3, 3),  padding = 'same', activation = 'relu') (inp)
pool_1 = MaxPooling2D(pool_size=2) (conv_1)
drop_1 = Dropout(0.25) (pool_1)
conv_2 = Convolution2D(filters = 64, kernel_size=(3, 3), padding = 'same', activation = 'relu') (drop_1)
conv_3 = Convolution2D(filters = 64, kernel_size=(3, 3), padding = 'same', activation = 'relu') (conv_2)
pool_2 = MaxPooling2D(pool_size=2) (conv_3)
drop_2 = Dropout(0.25) (pool_2)
conv_4 = Convolution2D(filters = 128, kernel_size=(3, 3), padding = 'same', activation = 'relu') (drop_2)
conv_5 = Convolution2D(filters = 128, kernel_size=(3, 3), padding = 'same', activation = 'relu') (conv_4)
pool_3 = MaxPooling2D(pool_size=2) (conv_5)
drop_3 = Dropout(0.25) (pool_3)

flat = Flatten()(drop_3)

net = Dense(500, activation = 'relu') (flat)

net = Dense(500, activation ='relu') (net)
'''
'''
#model 4 - .30 RESNET

'''
out_1 = Dense(11, activation='softmax', name = "output_font")(flat) 
out_2 = Dense(94, activation='softmax', name = "output_char")(flat) 
out_3 = Dense(2, activation='softmax', name = "output_bold")(flat) 
out_4 = Dense(2, activation='softmax', name = "output_italics")(flat) 

#define the model
model = Model(inputs=inp, outputs=[out_1, out_2, out_3, out_4])
model.compile(loss = ['categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy','binary_crossentropy'],optimizer='adam', metrics = ['accuracy'])

#saving the model summary
with open('../data_out/model/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

#training the model
#model.fit(X_train, Y_train, validation_split=0.2, epochs = 10, validation_data = (X_test, Y_test))
model.fit(X_train, Y_train, epochs = 5)

#print('Loss, Precision: ',model.evaluate(X_test, Y_test))  # Evaluate the trained model on the test set!

# serialize model to JSON
mm.save_model(model)

#%%

