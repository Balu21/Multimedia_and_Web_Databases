import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import NMF
from skimage import data, exposure
from scipy import spatial
import os
import math
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#Dataset directory
directory_in_str = 'Dataset2'
directory = directory_in_str

#Meta data info
HandInfo = pd.read_csv("HandInfo.csv")

#Gets subject IDs of all the images in the given dataset
subs = []
for file in os.listdir(directory):
    #print(file)
    subs.append((HandInfo.index[HandInfo['imageName'] == file]).tolist())
    
#print(np.array(subs).flatten())
#print(subs)
x = np.array(subs).flatten()
#print(x)
subID = []

#subjectID contains unique IDs from the Dataset
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

#Get HOG vectors for all the images in the dataset
for filename in os.listdir(directory):
    fnames.append(filename)
    img2 = Image.open(os.path.join(directory,filename))
    h, w = img2.size
	#Downscale images from 10 to 1
    img2 = img2.resize((int(h/10),int(w/10)), Image.ANTIALIAS)
    img2.save('sompicX.jpg') 

    image2 = Image.open('sompicX.jpg')

    fd2, hog_image2 = hog(image2, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm = 'L2-Hys', visualize=True, feature_vector = True, multichannel=True)

    fdout.append(fd2)

np.shape(fdout)


#Get SVD vectors from the HOG vectors
u, s, vh = np.linalg.svd(fdout, full_matrices=False)
#k = 40 and k = 30 are good
k = 30
temp2 = u[:,0:k]

#np.shape(temp2)

siz = len(subjectID)

#dist6 contains the similarity values for each subject ID to every other subjectID
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
                dist6[i][j] += abs(cosim[0][0])
        dist6[i][j] /= (len(index1)*len(index2))
        
k = int(sys.argv[1])

#k = 20

#Perform NMF on the subject-subject similarity matrix(dist6)
model = NMF(n_components = k, init = 'random', random_state = 0)
W = model.fit_transform(dist6)
H = model.components_        

#for i in range(k):
#    print(H[i].argmax() , H[i][H[i].argmax()])

output = []

#Output the top k latent semantics
for i in range(k):
    out = []
    temp1 = H[i].argsort()
    t1 = temp1.tolist()[::-1]
    temp2 = H[i].tolist()
    temp2.sort(reverse = True)
    for j in range(len(H[0])):
        out.append((subjectID[t1[j]], round(temp2[j], 3)))
    output.append(out)

#print(output[0])

with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(output)
    
print('\n\t Open "out.csv" file to view the output\n')