import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import NMF
from skimage import data, exposure
import os
import math
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.spatial import distance

#training_data = train
header = ["f1", "f2", "f3", "f4","f5", "f6", "f7", "f8", "f9", "f10", "label"]


def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    counts = {}  
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))
    
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):

    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)
    
    return Decision_Node(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def accuracy(trueValues, predictedValues):
    pred = []
    count = 0
    for p in predictedValues:
        for key in p.keys():
            pred.append(key)
    
    for i in range(len(trueValues)):
        if trueValues[i] == pred[i]:
            count += 1
    
    return float(count)*100/len(trueValues)

import os
import math
import numpy as np
import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from scipy.integrate import quad
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy import spatial
from pathlib import Path
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from io import BytesIO
#from IPython.display import HTML
import glob
import base64

def gaussian(x):
    mu = 0
    sigma = 0.25
    k = 1 / (sigma * math.sqrt(2*math.pi))
    s = -1.0 / (2 * sigma * sigma)

    return k * math.exp(s * (x - mu)*(x - mu))

def GaussProb_calc(x, y):
    Denominator, derr = quad(gaussian, 0, 1)
    Numerator, nerr = quad(gaussian, x, y)
    return Numerator/Denominator

def GaussianBand(d):
    Band = np.full(d, -1.0, dtype="float64")
    for index in range(len(Band)):
        Band[index] = GaussProb_calc(index/d, (index+1)/d)
    np.savetxt("bandValues.txt", Band)
    return Band

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

print("Enter the number of Layer")
L = int(input())
print("Enter the number of hashes per layer")
K = int(input())
imageId='Hand_0000674.jpg'
print("Enter the value of t")
t = int(input())

w = 4
b = np.full(shape=(L, K), fill_value=-1)
# read the vector inputs from too

randomVectors_List = []
# fetch the image - feature matrix
imagetfidf = np.load("object_latent_svd.npy")
imagetfidf=imagetfidf.T
data_matrix=imagetfidf
imagetfidf = np.divide(imagetfidf,np.sum(imagetfidf,axis=1,keepdims=True))

CountOfObjects, CountOfDimentions = imagetfidf.shape

# get the randon vectors
Band = GaussianBand(CountOfDimentions)

# create the Index structure
hash_LSH = np.full(shape=(L, K, CountOfObjects),
                   fill_value=-1, dtype="float64")
Q_hash = np.full(shape=(L, K), fill_value=-1, dtype="float64")

#band_index = 0
for i in range(L):
    for j in range(K):
        randomVector = np.asarray(
            Band[np.random.permutation(CountOfDimentions)])
        #np.savetxt("Output/randVect.txt", randomVector)
        randomVectors_List.append(randomVector)
        b[i][j] = np.random.randint(w+1)
        for k in range(CountOfObjects):
            hash_LSH[i][j][k] = float(
                np.dot(randomVector.T, imagetfidf[k]) + float(b[i][j]))/float(w)

imageIds_List = pd.read_csv("HandInfo.csv") 
image_names=imageIds_List['imageName'].values.tolist()
indexQImgId = image_names.index(imageId)


for i in range(L):
    for j in range(K):
        Q_hash[i][j] = hash_LSH[i][j][indexQImgId]

# normalize that data to above 10^8 to place them in different bins
for i in range(L):
    for j in range(K):
        #set it to 6
        power = 6
        hash_LSH[i][j] = np.floor(hash_LSH[i][j] * math.pow(10, power))
        Q_hash[i][j] = np.floor(Q_hash[i][j] * math.pow(10, power))

# Match the queryImage with the Hash_LSH

bucket = set()

for i in range(L):
    for j in range(K):
        for k in range(CountOfObjects):
            if hash_LSH[i][j][k] == Q_hash[i][j]:
                bucket.add(k)

LshImageIdList = []
for imageIndex in bucket:
    LshImageIdList.append(str(image_names[imageIndex]))
LshImageIdList.sort()
#print(len(LshImageIdList))


data_feature_matrix=data_matrix[list(bucket),:]

query_image=data_matrix[indexQImgId].reshape(1,-1)
query_image.shape


save_dict = {}

for i in bucket:
    compare_mat=data_matrix[i,:].reshape(1,-1)
    currVal=1 - spatial.distance.cosine(query_image, compare_mat)
    save_dict[image_names[i]] = currVal


count=0

for key, value in sorted(save_dict.items(), key=lambda item: item[1],reverse=True):
    if(count>t):
        break
    #print(key+"\t"+str(value))
    #data_uri=base64.b64encode(open('C:/Users/svasud13.ASURITE/Downloads/Hands/Hands/'+key,'rb').read()).decode('utf-8')
    #img_tag = '<img src="data:image/jpg;base64,{0}">'.format(data_uri)
    #10
    # print(img_tag)
    count=count+1


count = 0
filenames = []
for key, value in sorted(save_dict.items(), key=lambda item: item[1],reverse=True):
    if(count>t):
        break
    filenames.append(key)
    count += 1


directory_in_str = 'Hands'
directory = directory_in_str


fdout = []
fnames = []
for filename in filenames:
    fnames.append(filename)
    img2 = Image.open(os.path.join(directory,filename))
    h, w = img2.size
    img2 = img2.resize((int(h/10),int(w/10)), Image.ANTIALIAS)
    img2.save('sompicX.jpg') 

    image2 = Image.open('sompicX.jpg')

    fd2, hog_image2 = hog(image2, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm = 'L2-Hys', visualize=True, feature_vector = True, multichannel=True)

    fdout.append(fd2)
    
#print(len(fdout))

from sklearn.decomposition import PCA

k = 10
pca = PCA(n_components = k)
features = pca.fit_transform(fdout)
temp = pca.components_



from sklearn.decomposition import PCA
from scipy.spatial import distance

dist = []
for i in range(1,21):
    dist.append(distance.euclidean(features[0], features[i]))
    
arr = np.array(dist)
#print(arr)

maxD = arr.argsort()[-5:][::-1]
minD = arr.argsort()[:5][::1]

features = features.tolist()
li = []

for i in range(1,len(features)):
    for m in minD:
        if i == m+1:
            li.append(features[i] + ['Relevant'])
    for n in maxD:
        if i == n+1:
            li.append(features[i] + ['Irrelevant'])

dtree = build_tree(li) 

predictions = []
for row in features:
    predictions.append(classify(row, dtree))

rel = []
irr = []
pred = []
count = 0
for p in predictions:
    for key in p.keys():
        pred.append(key)
            
for i in range(len(pred)):
    if pred[i] == 'Relevant':
        rel.append(filenames[i])
    if pred[i] == 'Irrelevant':
        irr.append(filenames[i])
#print(rel)


import matplotlib.pyplot as plt

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

if(len(rel)>7):
    ran = 7
else:
    ran = len(rel)

if(len(irr)>7):
    ran2 = 7
else:
    ran2 = len(irr)

print("Relevant")
for i in range(ran):
	print(rel[i])
	
print("\nIrrelevant")
for i in range(ran2):
	print(irr[i])    


figures.append({'im'+str(i): plt.imread('Hands/'+rel[i]) for i in range(ran)})
figures.append({'im'+str(i): plt.imread('Hands/'+irr[i]) for i in range(ran2)})

# plot of the images in a figure, with 2 rows and 3 columns
plot_figures(figures, 2, 7)