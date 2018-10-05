
import numpy as np
from tqdm import tqdm
#import data_augmentation_module as dam
from keras.preprocessing.image import ImageDataGenerator


#this loads images from the npy file genereted by the data_generator.py 
data = np.load('../data/data_generator_matrix_representations.npy')
data = list(data)
#this loads the ground truth
gt = np.load('../data/ground_truth.npy')
gt = list(gt)

#our model will deal with scaned images that represent text. Since the input image to classify will rarely be perfectly defined, we are going to 
#perform some transformation on the pictures initially generated trying to mimic real world deformations.
#       1. 

gen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=29,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range =0.5,
    #horizontal_flip = True,
    #vertical_flip = True
    )

gen.fit(data)

count = 95
n = len(data)
print('\nAugmenting the dataset be patient... \n')
#for each image in tha dataset
for i in tqdm(range(n)):
    img = data[i]
    aug_iter = gen.flow(np.expand_dims(img, 0))
    
    #generate 10 samples of augmented images for the current image
    aug_images = [next(aug_iter)[0].astype(np.float64) for _ in range(20)]
    
    #adding the new picture in the data and in the ground truth
    for im in aug_images: data.append(im)
    for _ in range(len(aug_images)): gt.append(gt[i])
    
    for j in range(len(aug_images)):
            pic = aug_images[j]
            #saving the image on the file system
            '''
            pic = pic*255
            pic = Image.fromarray(pic.reshape(64, 64))
            pic = pic.convert('RGB')
            pic.save('../data/data_generator_images/img_'+str(count)+'_'+gt[i][0]+'augm_'+str(j)+'.png', 'PNG')
            '''
    
    count += 1
    
data = np.asarray(data)
gt = np.asarray(gt)
#save the data containing the images as np arrays into a npy file
outfile = '../data/data_aumentation_matrix_representations.npy'
np.save(outfile, data)
#save the data containing the ground truth. this matrix follows the same order as the previous one
outfile = '../data/ground_truth_data_augmentation.npy'
np.save(outfile, gt)

print('\nDone.\n')

#TOTAL_IMAGES = (2622 * 10) +2622 = 28 842