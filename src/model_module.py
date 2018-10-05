
#%% This function creates labels for the Fonts and Characters and update the Dataset replacing them by those numbers

def create_labels(data_y):
    font_names = list(set(data_y[:,0]))
    char_names = list(set(data_y[:,1]))
    
    #font name encoding we will convert the names into number,
    #so that they can be encoded later into a OneHotEncoded format
    font_to_int = dict((font, i) for i, font in enumerate(font_names))
    int_to_font = dict((i, font) for i, font in enumerate(font_names))
    
    char_to_int = dict((char, i) for i, char in enumerate(char_names))
    int_to_char = dict((i, char) for i, char in enumerate(char_names))
    
    #changing names to int
    for (i, font) in enumerate(data_y[:,0]): data_y[i,0] = font_to_int[font]
    for (i, char) in enumerate(data_y[:,1]): data_y[i,1] = char_to_int[char]
        
    return (font_to_int, char_to_int, int_to_font, int_to_char)

#%% this function saves the module into a json file and model weights into an h5 file

def save_model(model):
    # serialize model to JSON
    with open("../data_out/model/model.json", "w") as json_file:
        json_file.write(model.to_json())
    # serialize weights to HDF5
    model.save_weights("../data_out/model/model.h5")
    
#%%
