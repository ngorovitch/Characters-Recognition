import sys
import pickle
import pandas as pd
import numpy as np
from keras.models import model_from_json

#read input parameters
path_to_data = sys.argv[1]
output_file_name = sys.argv[2]

#load the input data
test_set = np.load(path_to_data)

#load the keras model
with open('../data_out/model/model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('../data_out/model/model.h5')

#make the predictions
Y_pred = model.predict(test_set)

#load the encoding data
font_encoding = pickle.load( open( "../data/font_encoding.p", "rb" ) )
inverted_font_encoding = pickle.load( open( "../data/inverted_font_encoding.p", "rb" ) )
char_encoding = pickle.load( open( "../data/char_encoding.p", "rb" ) )
inverted_char_encoding = pickle.load( open( "../data/inverted_char_encoding.p", "rb" ) )

#decoding the output
n = len(Y_pred[0])
Y_pred[0] = list(Y_pred[0])
Y_pred[1] = list(Y_pred[1])
Y_pred[2] = list(Y_pred[2])
Y_pred[3] = list(Y_pred[3])
for i in range(n): Y_pred[0][i] = list(Y_pred[0][i]).index(Y_pred[0][i].max())
for i in range(n): Y_pred[1][i] = list(Y_pred[1][i]).index(Y_pred[1][i].max())
for i in range(n): Y_pred[2][i] = list(Y_pred[2][i]).index(Y_pred[2][i].max())
for i in range(n): Y_pred[3][i] = list(Y_pred[3][i]).index(Y_pred[3][i].max())

#constructing the output
output = []
for i in range(n):
    row = {}
    row['char'] = inverted_char_encoding[Y_pred[1][i]]
    row['font'] = inverted_font_encoding[Y_pred[0][i]]
    row['bold'] = Y_pred[2][i]
    row['italics'] = Y_pred[3][i]
    output.append(row)
#converting list of dicts to panda dataframe to make the csv writing easier
output = pd.DataFrame(output)

#reorder the pandas columns
cols = ['char', 'font', 'bold', 'italics']
output = output[cols]

#writing the output into the csv file
output.to_csv('../data_out/test/'+output_file_name+'.csv', sep=',', encoding='utf-8', index=False, header=False)



