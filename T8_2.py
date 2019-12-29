import csv
import numpy as np
import pandas as pd
import sys
import os
from sklearn import preprocessing

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from io import BytesIO
from IPython.display import HTML
import glob
import base64
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation, NMF



def get_thumbnail(names,title):
    pth = names[0]
    name = names[1]
    wt = names[2]
    i = Image.open(pth)
    w, h = i.size
    draw = ImageDraw.Draw(i)
    font = ImageFont.truetype("arial.ttf", 100)
    text_w, text_h = draw.textsize(name, font)
    x_pos = 0#h#0#h - text_h
    y_pos = w//2+250
    ImageDraw.Draw(i).text((x_pos,y_pos),name,(0,0,0),font = font)
    x_pos =0 # h
    y_pos = 0#w//2-7
    ImageDraw.Draw(i).text((x_pos,y_pos),title+':'+wt[:7],(0,0,0),font = font)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}"></img>'

def data_visualizer(d = None,u=None,v_trans=None, db='testing',out = 'data-latentsemantics-visualizer.html'):   
    '''
    d - data matrix of dimensions n x m  ; n - number of objects ; m - number of features
    u - data latent semantic matrix of dimension n x k ; n - number of features; k - number of latent semantics
    v - feature latent semantic matrix of dimension m x k ; k - number of latent semantics ; m - number of features
    v_trans - transpose(v)
    db - folder of images 
    out - saves the to html file with name out
    '''
    k = u.shape[1]
    # sort across each latent semantic based on the max component in descending order find the corresponding indices
    ind = np.argsort(-1*u, axis=0)
    names = ind.ravel()
    pd.set_option('display.max_colwidth', -1)
    # extracting .jpg files from db
    image_ids = glob.glob(db+'/*.jpg')
    cols = ['latent semantic-'+ str(i+1) for i in range(k)]
    df_ind = pd.DataFrame(ind,columns = cols)
    #mapping the file names to the sorted indices
    for j in range(k):
        for i in range(len(df_ind['latent semantic-'+ str(j+1)])):
            df_ind.iloc[i,j] = image_ids[df_ind.iloc[i,j]]+'-'+image_ids[df_ind.iloc[i,j]]+'-'+str(u[df_ind.iloc[i,j]][j])
    #converting each image file name to thumbnails for displaying in HTML format
    for j in range(k):
        for i in range(len(df_ind['latent semantic-'+ str(j+1)])):
            df_ind.iloc[i,j] = get_thumbnail(df_ind.iloc[i,j].split('-'),'wt')
    formatters  = {'latent semantic-'+ str(i+1):image_formatter for i in range(k)}
    df_ind.to_html(out,formatters = formatters, escape=False,index = False)
    print('Data latent semantics are saved in '+out)
    return df_ind
    
def feature_visualizer(d = None, u = None, v_trans = None, db = 'testing',out = 'feature-latentsemantics-visualizer.html'):   
    '''
    d - data matrix of dimensions n x m  ; n - number of objects ; m - number of features
    u - data latent semantic matrix of dimension n x k ; n - number of features; k - number of latent semantics
    v - feature latent semantic matrix of dimension m x k ; k - number of latent semantics ; m - number of features
    v_trans - transpose(v)
    db - folder of images 
    out - saves the output to html file with name out
    '''
    v= np.transpose(v_trans)
    k = u.shape[1]
    dv = np.dot(d,v)
    ind = np.argmax(dv,axis =0)
    ind = np.reshape(ind,(1,k))
    image_ids = glob.glob(db+'/*.jpg')
    cols = ['latent semantic-'+ str(i+1) for i in range(k)]
    df_ind = pd.DataFrame(ind,columns = cols)
    
    #mapping the file names to the sorted indices
    for j in range(k):
        for i in range(len(df_ind['latent semantic-'+ str(j+1)])):
            df_ind.iloc[i,j] = image_ids[df_ind.iloc[i,j]]+'-'+image_ids[df_ind.iloc[i,j]]+'-'+str(dv[df_ind.iloc[i,j]][j])
            
    #converting each image file name to thumbnails for displaying in HTML format
    for j in range(k):
        for i in range(len(df_ind['latent semantic-'+ str(j+1)])):
            df_ind.iloc[i,j] = get_thumbnail(df_ind.iloc[i,j].split('-'),'scr')
    formatters  = {'latent semantic-'+ str(i+1):image_formatter for i in range(k)}
    df_ind.to_html(out,formatters = formatters, escape=False,index = False)
    print('Feature latent semantics are saved in '+out)
    return df_ind





HandInfo = pd.read_csv("HandInfo.csv")

#print(HandInfo.aspectOfHand.unique())
#left-hand, right-hand, dorsal, palmar, with accessories, without accessories, male, female

#Dataset directory
directory_in_str = 'Dataset2'
directory = directory_in_str

#Get all filenames from the dataset
filenames = []
for file in os.listdir(directory):
    filenames.append(file)

#Build a meta data dataframe
imgMeta = pd.DataFrame(columns = ['leftHand', 'rightHand', 'dorsal', 'palmar',\
                                     'withAccessories', 'withoutAccessories', 'male', 'female'])

#Get the values for the dataframe by refering to the actual meta data file (HandInfo.csv)
for i in range(len(filenames)):
    lh = 0
    rh = 0
    d = 0
    p = 0
    w = 0
    wo = 0
    m = 0
    f = 0
    index = np.array(HandInfo.index[HandInfo['imageName'] == filenames[i]])
    #print(index)
    if HandInfo.at[index[0], 'aspectOfHand'] == 'dorsal left':
        lh = 1
        d = 1
    if HandInfo.at[index[0], 'aspectOfHand'] == 'dorsal right':
        rh = 1
        d = 1
    if HandInfo.at[index[0], 'aspectOfHand'] == 'palmar left':
        lh = 1
        p = 1
    if HandInfo.at[index[0], 'aspectOfHand'] == 'palmar right':
        rh = 1
        p = 1
    if HandInfo.at[index[0], 'accessories'] > 0:
        w = 1
    else:
        wo = 1
    if HandInfo.at[index[0], 'gender'] == 'male':
        m = 1
    else:
        f = 1
    
    imgMeta.loc[i] = [lh] + [rh] + [d] + [p] + [w] + [wo] + [m] + [f]


#print(imgMeta.shape)

from sklearn.decomposition import NMF

k = int(sys.argv[1])
#k = 6
#Perform NMF on the image metadata matrix
model = NMF(n_components = k, init = 'random', random_state = 0)
W = model.fit_transform(np.array(imgMeta))
H = model.components_        

data_matrix = imgMeta
df_data = data_visualizer(d=data_matrix, u = W , v_trans = H, db = 'Dataset2')
df_feature = feature_visualizer(d=data_matrix, u = W , v_trans = H, db = 'Dataset2')

# Call the visualizer function for W and H matices