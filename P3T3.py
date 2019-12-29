import pandas as pd
import os
import numpy as np
import pickle
import cv2
from T1 import k_latent_semantics
from collections import defaultdict
from scipy import sparse
from scipy.spatial import distance
from flask import Flask, render_template
import matplotlib.pyplot as plt


app = Flask(__name__)

top_images = []

''' Function to Visualize the top 'm' similar images'''
def plot_figures(figures, nrows = 1, ncols=1):

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()

def get_top_images(scores, n, image_ids, folder_name):
    indices = np.argsort(scores)[::-1][:n].tolist()
    image_ids = list(image_ids)
    top_image_ids = []
    for index in indices:
        top_images.append(folder_name+"/" + image_ids[index] + '.jpg')
        top_image_ids.append(image_ids[index])
    figures = {}
    K = 1
    for image in top_images:
        figures['Similar-Image' + str(K) + ":" +  top_image_ids[K-1]] = plt.imread(image)
        K += 1
    return figures


def pagerank(graph, seed,image_ids, alpha=0.85, maxerr=0.001):
    graphsum = graph.sum(axis=0)
    graphsum[graphsum == 0] = 1
    graph /= graphsum
    data = sparse.csr_matrix(graph)
    n = len(image_ids)
    ro, r = np.zeros(n), np.ones(n)
    r = r/n
    # Setting teleport vector values
    teleport = np.zeros(n)
    for each in seed:
        teleport[each] = 1
    teleport = teleport / sum(teleport)
    while np.sum(np.abs(r - ro)) > maxerr :
        ro = r.copy()
        for i in range(n):
            links = np.array(data[:, i].todense())[:, 0]
            r[i] = ro.dot(links * alpha) + teleport[i] * (1 - alpha)
    return r / sum(r)


def compute_seed(img_list, image_ids):
    seed_vector = []
    image_ids = list(image_ids)
    for each in img_list:
        index = image_ids.index(each)
        seed_vector.append(index)

    return seed_vector

# @app.route("/")
# def visualize_images():
#     return render_template('ppr.html', top_images=top_images)



def create_graph(simi_matrix, k, image_ids):

    image_graph = defaultdict(list)
    for i in range(len(simi_matrix)):
        top_k = np.argsort(simi_matrix[i])[:k+1].tolist()

        sim_images_list = []
        for each in top_k:
            if each != i:
                sim_images_list.append(image_ids[each])

        image_graph[image_ids[i]] = sim_images_list

        for j in range(len(simi_matrix)):
            if j in top_k and j != i:
                simi_matrix[i, j] = 1
            else:
                simi_matrix[i, j] = 0

    img_graph = sparse.csr_matrix(simi_matrix, dtype=np.float)
    return img_graph


def find_k_most_similiarity(folder_name, k=5, K=10):
    image_comp, feature_vector = k_latent_semantics(folder_name, color_model='HOG', drt='PCA', k=30)
    folder_directory, file_directory = os.path.split(folder_name)
    feature_file = os.path.join(folder_directory, file_directory+"_HOG_PCA_30.csv")
    image_matrix = pd.read_csv(feature_file).values
    image_ids = image_matrix[:, 0]
    image_matrix = image_matrix[:, 1:]
    simi_matrix = distance.cdist(image_matrix, image_matrix, 'cosine')
    image_similarity_graph = create_graph(simi_matrix, k, image_ids)
    images_li = ['Hand_0008333', 'Hand_0006183', 'Hand_0000074']
    seed = compute_seed(images_li, image_ids)
    pr = pagerank(image_similarity_graph, seed, image_ids)
    figures = get_top_images(pr, K, image_ids, folder_name)
    return figures



if __name__ == '__main__':

    # k = int(input('Please enter the value of k'))
    # folder_name = input('please enter the folder name')
    figures = find_k_most_similiarity('phase3_sample_data/Labelled/Set2', 5, 10)
    plot_figures(figures,5,2)
