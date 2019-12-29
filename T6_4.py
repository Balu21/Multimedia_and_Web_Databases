import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from skimage import data, exposure
import os
import math
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

directory_in_str = str(sys.argv[1])
directory = directory_in_str

HandInfo = pd.read_csv("HandInfo.csv")

subs = []
for file in os.listdir(directory):
    #print(file)
    subs.append((HandInfo.index[HandInfo['imageName'] == file]).tolist())
    
#print(np.array(subs).flatten())
#print(subs)
x = np.array(subs).flatten()
#print(x)
subID = []

for i in x:
    subID.append(HandInfo.at[i, 'id'])
#print(np.array(subID))
w = np.array(subID).flatten()
#subjectID = np.unique(w)
indexes = np.unique(w, return_index=True)[1]
subjectID = [w[index] for index in sorted(indexes)]
#print(subjectID)

fdout = []
fnames = []
for filename in os.listdir(directory):
    fnames.append(filename)
    img2 = Image.open(os.path.join(directory,filename))
    h, w = img2.size
    img2 = img2.resize((int(h/10),int(w/10)), Image.ANTIALIAS)
    img2.save('sompicX.jpg') 

    image2 = Image.open('sompicX.jpg')

    fd2, hog_image2 = hog(image2, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm = 'L2-Hys', visualize=True, feature_vector = True, multichannel=True)

    fdout.append(fd2)

np.shape(fdout)

#index1 = [i for i, x in enumerate(subID) if x == 27]
#index2 = [i for i, x in enumerate(subID) if x == 55]
#print(index1)
#print(index2)

u, s, vh = np.linalg.svd(fdout, full_matrices=False)
#k = 40 and k = 30 are good
k = 30
temp2 = u[:,0:k]

#np.shape(temp2)

siz = len(subjectID)

dist6 = [[0 for p in range(siz)] for q in range(siz)]
for i in range(siz):
    for j in range(siz):
        index1 = [p for p, x in enumerate(subID) if x == subjectID[i]]
        #print(subjectID[i])
        index2 = [q for q, x in enumerate(subID) if x == subjectID[j]]
        #print(subjectID[j])
        for a in index1:
            for b in index2:
                aa = temp2[a].reshape(1,len(temp2[a]))
                bb = temp2[b].reshape(1,len(temp2[b]))
                cosim = cosine_similarity(aa, bb)
                dist6[i][j] += ((cosim[0][0] + 1)/2)
        dist6[i][j] /= (len(index1)*len(index2))
        

#print(dist6[0])

inp = int(sys.argv[2])
#inp = 27
index_value = subjectID.index(inp)


h = np.argsort(dist6[index_value])[::-1][:4]
#print(h)

filenames = []
imgcount = []
for i in h:
    li = [p for p, x in enumerate(subID) if x == subjectID[i]]
    #print(li)
    imgcount.append(len(li))
    for j in range(len(li)):
        fname = fnames[li[j]]
        filenames.append(fname)
    
def plot_figures(figures, nrows = 1, ncols=1):
    img = np.zeros([100,100,3],dtype=np.uint8)
    img.fill(255) 
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for j in range(nrows):
        
        for ind,title in enumerate(figures[j]):
            axeslist.ravel()[ind + j*ncols].imshow(figures[j][title], cmap=plt.gray())
            axeslist.ravel()[ind + j*ncols].set_title(title)
            axeslist.ravel()[ind + j*ncols].set_axis_off()
        if ncols > len(figures[j]):
            for i in range(len(figures[j]),max(imgcount)):
                axeslist.ravel()[i + j*ncols].imshow(img, cmap=plt.gray())
                axeslist.ravel()[i + j*ncols].set_title('')
                axeslist.ravel()[i + j*ncols].set_axis_off()
    plt.tight_layout()
    plt.show()

#number_of_im = 20

figures = []

figures.append({'im'+str(i): plt.imread(str(sys.argv[1])+'/'+filenames[i]) for i in range(imgcount[0])})
figures.append({'im'+str(i): plt.imread('Dataset2/'+filenames[i]) for i in range(imgcount[0],imgcount[0]+imgcount[1])})
figures.append({'im'+str(i): plt.imread('Dataset2/'+filenames[i]) for i in range(imgcount[0]+imgcount[1],imgcount[0]+imgcount[1]+imgcount[2])})
figures.append({'im'+str(i): plt.imread('Dataset2/'+filenames[i]) for i in range(imgcount[0]+imgcount[1]+imgcount[2],imgcount[0]+imgcount[1]+imgcount[2]+imgcount[3])})

# plot of the images in a figure, with 2 rows and 3 columns
plot_figures(figures, 4, max(imgcount))