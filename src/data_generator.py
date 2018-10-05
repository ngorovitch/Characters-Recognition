
import numpy as np
import os
import string
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


#this is the future numpy array
data = []

# all the possible caracters
all_characters = string.printable[:-6]

# ground truth
gt = []

#path containing the fonts
fonts_path = '../data/fonts/'

print('\ngenerating all the possible combinations of font/character/bold/italic images and the matrix representations ... \n')
#for each font in fonts_path
for font_name in tqdm(os.listdir(fonts_path)):
    
    if font_name.endswith(".ttf") or  font_name.endswith(".otf"): 
        # to include in the the images names
        num = 1
    
        for character in all_characters:      
            
            '''
            #this should check if the the font supports the current character  but it's too slow
            ttf = TTFont(fonts_path + font_name, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
            chars = [x[0] for x in list(chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables))]
            if ord(character) in chars:
            '''
            path = '../data/fonts/' + font_name
            font = ImageFont.truetype(font=path, size=30)
            
            # create a blank canvas with extra space on the margins
            canvas = Image.new('RGB', (64, 64), (255, 255, 255))
            
            # draw the text onto the text canvas, use black as the text color and center the text on the image 
            draw = ImageDraw.Draw(canvas)
            cw, ch = draw.textsize(character, font=font)
            draw.text(((64 - cw)/2, ((64-ch)/2)-5), character, font = font, fill = "#000000")
            
            extrema = canvas.convert("L").getextrema()
            #this checks if the image is blank which much more faster than the previous method for checking if font supports a character
            if extrema != (255, 255):
                #saving the image on the file system
                '''
                if not os.path.exists('../data/data_generator_images'):
                    os.makedirs('../data/data_generator_images')
                
                img_name = '../data/data_generator_images/img_'+str(num)+'_'+font_name[:-4]+'.png'
                canvas.save(img_name, 'PNG')
                '''
                num += 1
                
                #convert the image into numpy array: the convert L is to use the black and white mode
                canvas_nparray = np.asarray(canvas.convert("L")).reshape((64, 64, 1))/255
                data.append(canvas_nparray)
                
                #inserting into the ground truth info about the current character
                font_info = font_name[:-4].split('_')
                # first case: no bold no italic
                if(len(font_info) == 1):
                    info = [font_info[0], character, 0, 0]
                
                # second case: no bold but italic
                if(len(font_info) == 2 and font_info[1] == 'I'):
                    info = [font_info[0], character, 0, 1]
                
                # tirth case: bold but no italic
                if(len(font_info) == 2 and font_info[1] == 'B'):
                    info = [font_info[0], character, 1, 0]
                    
                # fourth case: bold and italic
                if(len(font_info) == 3):
                    info = [font_info[0], character, 1, 1]
                    
                gt.append(info)
            
        
#convert the images list and the ground truth list into numpy arrays
data = np.asarray(data)
gt = np.asarray(gt)
#save the data containing the images as np arrays into a npy file
outfile = '../data/data_generator_matrix_representations.npy'
np.save(outfile, data)
#save the data containing the ground truth. this matrix follows the same order as the previous one
outfile = '../data/ground_truth.npy'
np.save(outfile, gt)
print('\nDone.\n')
