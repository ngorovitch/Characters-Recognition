import numpy as np
from PIL import Image
#import imutils
#import cv2
import math
import tensorflow as tf
import random

#%% this function turns into white the corners of a 64*64 picture

def white_corners(img_as_matrix):
    #removing the black corners
    #starting from left
    for r in range(64):
        for c in range(64): 
            if(img_as_matrix[r, c]) == 1:
                break
            else:
                img_as_matrix[r, c] = 1
    
    #starting from right
    r = 0;
    while r < 64:
        c = 63
        while c >= 0: 
            if(img_as_matrix[r, c]) == 1:
                break
            else:
                img_as_matrix[r, c] = 1
            c -= 1
        r += 1

#%% this function performs a rotation of a given angle of a 64*64 picture

def rotate_and_save(img_indice, all_images, ground_T, count_for_saving, angle):
    img = all_images[img_indice]
    #rotated = imutils.rotate_bound(img, angle)
    #rotated = ((cv2.resize(rotated,(64, 64))).reshape(64, 64, 1))
    rotated = Image.fromarray((img.reshape(64, 64))*255)
    rotated = rotated.rotate(angle, resample=Image.BICUBIC)
    rotated = np.asarray(rotated).reshape((64, 64, 1))/255
    
    #removing black corners
    white_corners(rotated)
    
    #adding the new picture in the data and in the ground truth
    all_images.append(rotated)
    ground_T.append(ground_T[img_indice])
    
    '''
    #saving the image on the file system
    rotated = rotated*255
    rot = Image.fromarray(rotated.reshape(64, 64))
    rot = rot.convert('RGB')
    rot.save('../data/data_generator_images/img_'+str(count_for_saving)+'_'+ground_T[img_indice][0]+'Rot'+str(angle)+'.png', 'PNG')
    '''
    return rotated/255

#%% this function performs a outward scaling of a 64*64 picture with a given outscaling percentage

def outward_scaling(img_indice, all_images, ground_T, count_for_saving, percentage):
    #let's zoom the image resizing it from 64*64 to new_dim*new_dim
    img = all_images[img_indice]
    new_dim = int((64*(percentage/100))+64)
    scaled_out = Image.fromarray((img.reshape(64, 64))*255)
    scaled_out = scaled_out.resize((new_dim, new_dim), Image.ANTIALIAS)
    #now let's crop the central part to go back to 64*64
    left = (new_dim - 64)/2
    top = (new_dim - 64)/2
    right = (new_dim + 64)/2
    bottom = (new_dim + 64)/2
    scaled_out = scaled_out.crop((left, top, right, bottom))
    
    #adding the new picture in the data and in the ground truth
    scaled_out_nparray = np.asarray(scaled_out).reshape((64, 64, 1))/255
    all_images.append(scaled_out_nparray)
    ground_T.append(ground_T[img_indice])
    #saving the image on the file system
    '''
    scaled_out = scaled_out.convert('RGB')
    scaled_out.save('../data/data_generator_images/img_'+str(count_for_saving)+'_'+ground_T[img_indice][0]+'Scaled_out+'+str(percentage)+'.png', 'PNG')
    '''

    return scaled_out_nparray


#%% this function performs a inward scaling of a 64*64 picture with a given inscaling percentage

def inward_scaling(img_indice, all_images, ground_T, count_for_saving, percentage):
    #let's scale in the image resizing it from 64*64 to 32*32
    img = all_images[img_indice]
    scaled_in = Image.fromarray((img.reshape(64, 64))*255)
    new_dim = math.ceil(64 - (64*(percentage/100)))
    scaled_in = scaled_in.resize((new_dim, new_dim), Image.ANTIALIAS) #ANTIALIAS is used to avoid losing quality
    
    #create a blank image
    canvas = Image.new('RGB', (64, 64), (255, 255, 255))
    canvas = canvas.convert("L")
    canvas_nparray = np.asarray(canvas)/255
    canvas_nparray = canvas_nparray*255 #this apperently useless operation is used to convert the 
                                        #type font int to float
    
    # insert the zoomed out image in the blank image
    canvas_nparray.setflags(write=1) #to avoid the read only error
    remain = 64 - new_dim
    start = math.ceil(remain/2)
    end = 64 - (remain - start)
    canvas_nparray[start:end, start:end] = np.asarray(scaled_in)
    canvas_nparray.setflags(write=0) #reset the matrix to read only
    
    #adding the new picture in the data and in the ground truth
    scaled_in_nparray = canvas_nparray.reshape((64, 64, 1))/255
    all_images.append(scaled_in_nparray)
    ground_T.append(ground_T[img_indice])
    #saving the image on the file system
    '''
    scaled_in = Image.fromarray(canvas_nparray).convert('RGB')
    scaled_in.save('../data/data_generator_images/img_'+str(count_for_saving)+'_'+ground_T[img_indice][0]+'Scaled_in-'+str(percentage)+'.png', 'PNG')
    '''
    
    return scaled_in_nparray

#%% this function performs a 15 pixels translation on a 64*64 image following the indicated direction

def translation(img_indice, all_images, ground_T, count_for_saving, direction = 'l'):
    img = all_images[img_indice]
    #[ [size of text,left bottom corner,from left to right], [right top corner,size of text,from top to down] ]
    if direction == 'l':
        #translation_matrix = np.float32([ [1,0,15], [0,1,0] ])
        #img_translation = cv2.warpAffine(img, translation_matrix, (64, 64))
        '''
        x = tf.image.pad_to_bounding_box(x, pad_top, pad_left, height + pad_bottom + pad_top, width + pad_right + pad_left)
        output = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, height, width)
        '''
        img = tf.convert_to_tensor(img, dtype = tf.float32)
        img = tf.image.pad_to_bounding_box(img, 0, 0, 64 + 0 + 0, 64 + 15 + 0)
        img_translation = tf.image.crop_to_bounding_box(img, 0, 15, 64, 64)
    if direction == 'r':
        #translation_matrix = np.float32([ [1,0,-15], [0,1,0] ])
        #img_translation = cv2.warpAffine(img, translation_matrix, (64, 64))
        img = tf.convert_to_tensor(img, dtype = tf.float32)
        img = tf.image.pad_to_bounding_box(img, 0, 15, 64 + 0 + 0, 64 + 0 + 15)
        img_translation = tf.image.crop_to_bounding_box(img, 0, 0, 64, 64)
    if direction == 'b':
        #translation_matrix = np.float32([ [1,0, 0], [0,1,15] ])
        #img_translation = cv2.warpAffine(img, translation_matrix, (64, 64))
        img = tf.convert_to_tensor(img, dtype = tf.float32)
        img = tf.image.pad_to_bounding_box(img, 15, 0, 64 + 0 + 15, 64 + 0 + 0)
        img_translation = tf.image.crop_to_bounding_box(img, 0, 0, 64, 64)
    if direction == 't':
        #translation_matrix = np.float32([ [1,0, 0], [0,1,-15] ])
        #img_translation = cv2.warpAffine(img, translation_matrix, (64, 64))
        img = tf.convert_to_tensor(img, dtype = tf.float32)
        img = tf.image.pad_to_bounding_box(img, 0, 0, 64 + 15 + 0, 64 + 0 + 0)
        img_translation = tf.image.crop_to_bounding_box(img, 15, 0, 64, 64)
    if direction == 'trc': #top right corner
        #translation_matrix = np.float32([ [1,0, 15], [0,1,-15] ])
        #img_translation = cv2.warpAffine(img, translation_matrix, (64, 64))
        img = tf.convert_to_tensor(img, dtype = tf.float32)
        img = tf.image.pad_to_bounding_box(img, 0, 15, 64 + 15 + 0, 64 + 0 + 15)
        img_translation = tf.image.crop_to_bounding_box(img, 15, 0, 64, 64)
    if direction == 'tlc': #top left corner
        #translation_matrix = np.float32([ [1,0, -15], [0,1,-15] ])
        #img_translation = cv2.warpAffine(img, translation_matrix, (64, 64))
        img = tf.convert_to_tensor(img, dtype = tf.float32)
        img = tf.image.pad_to_bounding_box(img, 0, 0, 64 + 15 + 0, 64 + 15 + 0)
        img_translation = tf.image.crop_to_bounding_box(img, 15, 15, 64, 64)
    if direction == 'brc':
        #translation_matrix = np.float32([ [1,0, 15], [0,1,15] ])
        #img_translation = cv2.warpAffine(img, translation_matrix, (64, 64))
        img = tf.convert_to_tensor(img, dtype = tf.float32)
        img = tf.image.pad_to_bounding_box(img, 15, 15, 64 + 0 + 15, 64 + 0 + 15)
        img_translation = tf.image.crop_to_bounding_box(img, 0, 0, 64, 64)
    if direction == 'blc':
        #translation_matrix = np.float32([ [1,0, -15], [0,1,15] ])
        #img_translation = cv2.warpAffine(img, translation_matrix, (64, 64))
        img = tf.convert_to_tensor(img, dtype = tf.float32)
        img = tf.image.pad_to_bounding_box(img, 15, 0, 64 + 0 + 15, 64 + 15 + 0)
        img_translation = tf.image.crop_to_bounding_box(img, 0, 15, 64, 64)
    
    sess = tf.Session()
    with sess.as_default():
        img_translation = img_translation.eval()
    white_corners(img_translation)
    
    #adding the new picture in the data and in the ground truth
    img_translate = img_translation.reshape((64, 64, 1))
    all_images.append(img_translate)
    ground_T.append(ground_T[img_indice])
    #saving the image on the file system
    '''
    img_translated = Image.fromarray((img_translation.reshape(64, 64))*255).convert('RGB')
    img_translated.save('../data/data_generator_images/img_'+str(count_for_saving)+'_'+ground_T[img_indice][0]+'Translated-'+direction+'.png', 'PNG')
    '''
    
    return img_translate

#%% this function add on a 64*64 picture gaussian noise with variance = 0.03 and mean = 0    

def add_gauss_noise(all_images, ground_T, count_for_saving, image=[], img_indice = 0, precedent_trasform=''):
    if image == []:
        image = all_images[img_indice]
    row,col,ch= image.shape
    mean = 0
    #var = 0.03
    var = random.uniform(0, 0.03)
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss

    #adding the new picture in the data and in the ground truth
    noised_img = noisy.reshape((64, 64, 1))
    all_images.append(noised_img)
    ground_T.append(ground_T[img_indice])
    #saving the image on the file system
    '''
    noised = Image.fromarray((noisy.reshape(64, 64))*255).convert('RGB')
    noised.save('../data/data_generator_images/img_'+str(count_for_saving)+'_'+ground_T[img_indice][0]+'Gaussian_noised'+precedent_trasform+'.png', 'PNG')
    '''
    return noised_img

#%% this function add on a 64*64 picture the gray effect to simulate the case when there is not enought ancher to well define the character

def gray(img_indice, all_images, ground_T, count_for_saving):
   
    img = all_images[img_indice]
    row,col,ch= img.shape
    #create a gray image
    canvas = Image.new('RGB', (64, 64), (192,192,192))
    canvas = canvas.convert("L")
    canvas_nparray = np.asarray(canvas)/255
    canvas_nparray = canvas_nparray.reshape(row,col,ch)
    #adding the new picture in the data and in the ground truth    
    gray = img + canvas_nparray
    gray_img = gray.reshape((64, 64, 1))
    all_images.append(gray_img)
    ground_T.append(ground_T[img_indice])
    #saving the image on the file system
    '''
    grayed = Image.fromarray((gray.reshape(64, 64))*255).convert('RGB')
    grayed.save('../data/data_generator_images/img_'+str(count_for_saving)+'_'+ground_T[img_indice][0]+'Grayed.png', 'PNG')
    '''
    
    return gray_img

#%%

    
    