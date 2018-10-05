
#%% importing routine libraries

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input,Convolution2D, BatchNormalization, concatenate
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
oh = OneHotEncoder()

#%% loading up the dataset(pictures) and the ground truth (labels)

data_x= np.load('../data/data_aumentation_matrix_representations.npy')
data_y= np.load('../data/ground_truth_data_augmentation.npy')


#%% Data preparation

#split the datasets in 
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=.3, train_size = .7)

# Loading the dependent data and transforming it into a onehotencoded format,
Y_train = [oh.fit_transform(Y_train[:,0:1]).todense(),
           oh.fit_transform(Y_train[:,1:2]).todense(),
           oh.fit_transform(Y_train[:,2:3]).todense(),
           oh.fit_transform(Y_train[:,3:4]).todense()]
Y_test = [oh.fit_transform(Y_test[:,0:1]).todense(),
          oh.fit_transform(Y_test[:,1:2]).todense(),
          oh.fit_transform(Y_test[:,2:3]).todense(),
          oh.fit_transform(Y_test[:,3:4]).todense(),]

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

#model 2 - .53
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


dense_2 = Dense(32, kernel_initializer='he_normal', name='dense2')(gru2_merged)
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
out_1 = Dense(11, activation='softmax', name = "output_font")(flat) 
out_2 = Dense(94, activation='softmax', name = "output_char")(flat) 
out_3 = Dense(2, activation='softmax', name = "output_bold")(flat) 
out_4 = Dense(2, activation='softmax', name = "output_italics")(flat) 

#define the model
model = Model(inputs=inp, outputs=[out_1, out_2, out_3, out_4])
model.compile(loss = ['categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy','binary_crossentropy'],optimizer='adam', metrics = ['accuracy'])

#training the model
model.fit(X_train, Y_train, epochs = 5)

print('Loss, Precision: ',model.evaluate(X_test, Y_test))  # Evaluate the trained model on the test set!

#confusion matrix
Y_pred_0= model.predict(X_test)

# serialize model to JSON
#mm.save_model(model)

#%%


#%%
font_test = [np.argmax(e) for e in Y_test[0]]
char_test = [np.argmax(e) for e in Y_test[1]]
bold_test = [np.argmax(e) for e in Y_test[2]]
italic_test = [np.argmax(e) for e in Y_test[3]]
t= list(zip(font_test,char_test,bold_test, italic_test))

font_pred = [np.argmax(e) for e in Y_pred_0[0]]
char_pred = [np.argmax(e) for e in Y_pred_0[1]]
bold_pred = [np.argmax(e) for e in Y_pred_0[2]]
italic_pred = [np.argmax(e) for e in Y_pred_0[3]]
result= list(zip(font_pred,char_pred,bold_pred, italic_pred))
#%%

#font accuracy
match =0
for i in range(len(font_test)):
    if font_pred[i] == font_test[i]:
        match+=1
fa = match/len(font_test)

#char accuracy
match =0
for i in range(len(char_test)):
    if char_pred[i] == char_test[i]:
        match+=1
ca = match/len(char_test)

#bold accuracy
match =0
for i in range(len(bold_test)):
    if bold_pred[i] == bold_test[i]:
        match+=1
ba = match/len(bold_test)

#italic accuracy
match =0
for i in range(len(italic_test)):
    if italic_pred[i] == italic_test[i]:
        match+=1
ia = match/len(italic_test)

partial_accuracy = 0.3*fa + 0.3*ca + 0.2*ba + 0.2*ia 

print('\nPartial_accuracy: '+ str(partial_accuracy))