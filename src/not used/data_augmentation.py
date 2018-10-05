
import numpy as np
from tqdm import tqdm
import data_augmentation_module as dam
from random import randint


#this loads images from the npy file genereted by the data_generator.py 
data = np.load('../data/data_generator_matrix_representations.npy')
data = list(data)
#this loads the ground truth
gt = np.load('../data/ground_truth.npy')
gt = list(gt)

#our model will deal with scaned images that represent text. Since the input image to classify will rarely be perfectly defined, we are going to 
#perform some transformation on the pictures initially generated trying to mimic real world deformations.
#       1. rotations between 5° AND 25° left and right to capture the case when the paper inclined while inserting it in the scanner.
#       2. outward scaling between 30 and 50%
#       3. inward scaling between 30 and 50%
#       4. the way the images are generated make them stand in the center of the picture so we are going to translate them to 
#          the right, to the left, to the top, to the bottom, and to all the 4 corners
#       5. adding some random Gaussian Noise to the images
#       6. adding a gray effect to mimic the scanner color effect


count = 95
n = len(data)
print('\nAugmenting the dataset be patient... \n')
#for each image in tha dataset
for i in tqdm(range(n)):
    #%% 2 rotations of angles random angles between 5° AND 25° left and right
        #they are also gaussian noised: total = 8
    
    a1 = randint(5, 25)
    a2 = randint(5, 25)
    a3 = randint(-25, -5)
    a4 = randint(-25, -5)
    #right side rotations
    #dam.rotate_and_save(i, data, gt, count, angle = a1)
    dam.add_gauss_noise(data, gt, count, image = dam.rotate_and_save(i, data, gt, count, angle = a1), img_indice = i, precedent_trasform = 'Rotation+'+str(a1))
    #dam.rotate_and_save(i, data, gt, count, angle = a2)
    dam.add_gauss_noise(data, gt, count, image = dam.rotate_and_save(i, data, gt, count, angle = a2), img_indice= i, precedent_trasform = 'Rotation+'+str(a2))
    #dam.rotate_and_save(i, data, gt, count, angle = a2)
    #left side rotations
    #dam.rotate_and_save(i, data, gt, count, angle = a3)
    dam.add_gauss_noise(data, gt, count, image = dam.rotate_and_save(i, data, gt, count, angle = a3), img_indice = i, precedent_trasform = 'Rotation'+str(a3))
    #dam.rotate_and_save(i, data, gt, count, angle = a3)
    dam.add_gauss_noise(data, gt, count, image = dam.rotate_and_save(i, data, gt, count, angle = a4), img_indice = i, precedent_trasform = 'Rotation'+str(a4))
    #dam.rotate_and_save(i, data, gt, count, angle = a4)
    #%%
    
    
    
    #%% outward scaling of 30 and 50%,
        #they are also gaussian noised: total = 4
    p1 = randint(30, 50)
    p2 = randint(30, 50)
    
    #dam.outward_scaling(i, data, gt, count, percentage = 10)
    dam.add_gauss_noise(data, gt, count, image = dam.outward_scaling(i, data, gt, count, percentage = p1), img_indice = i, precedent_trasform = 'Outward_scaling+'+str(p1))
    #dam.outward_scaling(i, data, gt, count, percentage = 30)
    dam.add_gauss_noise(data, gt, count, image = dam.outward_scaling(i, data, gt, count, percentage = p2), img_indice = i, precedent_trasform = 'Outward_scaling+'+str(p2))
    #dam.outward_scaling(i, data, gt, count, percentage = 50)
    
    #%%
    
    
    #%% inward scaling of 30 and 50%: total = 4
    p3 = randint(30, 50)
    p4 = randint(30, 50)
    
    dam.add_gauss_noise(data, gt, count, image = dam.inward_scaling(i, data, gt, count, percentage = p3), img_indice = i, precedent_trasform = 'Inward_scaling+'+str(p3))
    #dam.inward_scaling(i, data, gt, count, percentage = 20)
    #dam.inward_scaling(i, data, gt, count, percentage = 30)
    #dam.inward_scaling(i, data, gt, count, percentage = 40)
    dam.add_gauss_noise(data, gt, count, image = dam.inward_scaling(i, data, gt, count, percentage = p4), img_indice = i, precedent_trasform = 'Inward_scaling+'+str(p4))
    
    #%%
    
    
    #%% translating image by 15 pixels to: left, right, top, bottom, top right corner, top left corner, bottom right corner, bottom left corner:
    # total: 8
    
    dam.translation(i, data, gt, count, direction = 'l')
    #dam.add_gauss_noise(data, gt, count, image = dam.translation(i, data, gt, count, direction = 'r'), img_indice = i, precedent_trasform = 'Translation_r')
    dam.translation(i, data, gt, count, direction = 'r')
    #dam.add_gauss_noise(data, gt, count, image = dam.translation(i, data, gt, count, direction = 't'), img_indice = i, precedent_trasform = 'Translation_t')
    dam.translation(i, data, gt, count, direction = 't')
    dam.translation(i, data, gt, count, direction = 'b')
    dam.translation(i, data, gt, count, direction = 'trc')
    dam.translation(i, data, gt, count, direction = 'tlc')
    #dam.add_gauss_noise(data, gt, count, image = dam.translation(i, data, gt, count, direction = 'brc'), img_indice = i, precedent_trasform = 'Translation_brc')
    dam.translation(i, data, gt, count, direction = 'brc')
    dam.translation(i, data, gt, count, direction = 'blc')
    
    #%%
    
    #%% adding gaussian noise to the images: total = 1
    
    dam.add_gauss_noise(data, gt, count, img_indice = i)
    
    #%%
    
    
    #%% adding gray effect: total = 1
    
    dam.gray(i, data, gt, count)
    
    #%%
    count += 1
    #TOTAL = 26
    #print('\nAugmented '+str(i+1)+'/'+str(n)+'...')
    

data = np.asarray(data)
gt = np.asarray(gt)
#save the data containing the images as np arrays into a npy file
outfile = '../data/data_aumentation_matrix_representations.npy'
np.save(outfile, data)
#save the data containing the ground truth. this matrix follows the same order as the previous one
outfile = '../data/ground_truth_data_augmentation.npy'
np.save(outfile, gt)

print('\nDone.\n')

#TOTAL_IMAGES = (2622 * 26) +2622 = 70 794
