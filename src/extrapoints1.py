
import numpy as np
from keras.models import model_from_json
from keras.models import Model
import matplotlib.pyplot as plt

print('\ngenerating intermediate output images for convolutional and pooling layers ... \n')

#load the input data
data = np.load('../data/data_aumentation_matrix_representations.npy')

#load the keras model
with open('../data_out/model/model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('../data_out/model/model.h5')



sample_image = data[100]
# Keras requires the image to be in 4D So we add an extra dimension to it.
img = np.expand_dims(sample_image, axis=0)


def intermediate_output(model, layer_index, img, name):
    #new model with intermediate output
    intermediate_model = Model(model.inputs, model.layers[layer_index].output)
    #set the same weights for the new model
    intermediate_model.set_weights(model.get_weights())
    
    #get the 16 output images
    output_images = intermediate_model.predict(img)[0]
    
    #save the output of each filter as png image
    nbr_filters = 16
    for f in range(nbr_filters):
        filter_output = output_images[:,:,f]
        plt.subplot(4,4,f+1)
        plt.imshow(filter_output, cmap='gray')
        plt.axis('off')
    plt.savefig('../data_out/img/output_'+str(name)+'.png')
    plt.close()

intermediate_output(model, 1, img, 'conv_1') #first convolutional layer
intermediate_output(model, 2, img, 'pool_1') #first pooling layer
intermediate_output(model, 3, img, 'conv_2') #2nd convolutional layer
intermediate_output(model, 4, img, 'pool_2') #2nd pooling layer

print('\nDone.\n')








































