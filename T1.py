from phase1combined import storing_features
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

'''Dimensionality reduction Methods which gives the object-latent Semantics and Feature-Latent Semantics'''

def pca(dataset, n_components):
    if n_components == None:
        n_components = min(dataset.shape[0], dataset.shape[1])
    pca_model = PCA(n_components=n_components)
    converted_matrix = pca_model.fit_transform(dataset)
    v_trans = pca_model.components_
    return pca_model, converted_matrix, v_trans


def SVD(dataset, n_components):
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    proj_data = svd.fit_transform(dataset)
    v_trans = svd.components_
    return svd, proj_data, v_trans


def LDA(dataset, n_components):
    lda = LatentDirichletAllocation(n_components=n_components)
    proj_data = lda.fit_transform(dataset)
    v_trans = lda.components_
    return lda, proj_data, v_trans


def nmf(dataset, n_components):
    if n_components == None:
        n_components = dataset.shape[1]
    nmf_model = NMF(n_components=n_components, init='random', random_state=0)
    proj_data = nmf_model.fit_transform(dataset)
    v_trans = nmf_model.components_
    return nmf_model, proj_data, v_trans

''' Function which gives the Object latent Semantics given a Color Model, 
    Dimensionality Reduction Technique and K'''

def k_latent_semantics(test_path, color_model, drt, k):

    base_directory = Path(__file__).parent

    folder = glob.glob(str(test_path)+"/*.jpg")

    ''' Calling the 'Storing Features' function if features matrix for a given color model
     do not exist'''

    if color_model == 'CM':
        f_name = test_path + "_CM_Features.csv"
        file_path = base_directory/f_name

        if not file_path.exists():
            storing_features(test_folderpath=test_path,model=color_model)

        data = pd.read_csv(f_name)
        data = data.values
        image_ids = data[:,0]
        data_matrix = data[:,1:]
        # min_values = np.min(data_matrix,axis=0)
        # for i in range(data_matrix.shape[1]):
        #     data_matrix[:,i] += abs(min_values[i])

    elif color_model == 'LBP':
        f_name = test_path + "_LBP_Features.csv"
        file_path = base_directory/f_name

        if not file_path.exists():
            storing_features(test_folderpath=test_path,model=color_model)

        data = pd.read_csv(f_name)
        data = data.values
        data_matrix = data[:,1:]
        image_ids = data[:,0]

    elif color_model == 'HOG':
        f_name = test_path + "_HOG_Features.csv"
        file_path = base_directory/f_name

        if not file_path.exists():
            storing_features(test_folderpath=test_path,model=color_model)

        data = pd.read_csv(f_name)
        data = data.values
        data_matrix = data[:,1:]
        image_ids = data[:,0]

    elif color_model == 'SIFT':
        f_name = test_path + "_SIFT_Features.csv"
        file_path = base_directory/f_name

        if not file_path.exists():
            storing_features(test_folderpath=test_path,model=color_model)

        data = pd.read_csv(f_name)
        data = data.values
        data_matrix = data[:,1:]
        image_ids = data[:,0]

    if drt == 'PCA':
        comp, proj_mat, feature_vec = pca(data_matrix, k)

    elif drt == 'SVD':
        comp, proj_mat, feature_vec  = SVD(data_matrix, k)

    elif drt == 'LDA':

        if color_model == 'CM':
            final_matrix = []
            for i in range(data_matrix.shape[0]):
                for j in range(0, data_matrix.shape[1], 9):
                    final_matrix.append(list(data_matrix[i, j:j+9]))
            final_matrix = np.array(final_matrix)
            kmeans = KMeans(n_clusters=400, random_state=0).fit(final_matrix)
            final_store_matrix = np.zeros((data_matrix.shape[0], 400))
            for i in range(data_matrix.shape[0]):
                start = i * data_matrix.shape[1]//9
                print(start)
                for j in range(data_matrix.shape[1]//9):
                    final_store_matrix[i, kmeans.labels_[start+j]] += 1
            data_matrix = final_store_matrix

        comp, proj_mat, feature_vec = LDA(data_matrix, k)

    elif drt == 'NMF':
        if color_model == 'CM':
            final_matrix = []
            for i in range(data_matrix.shape[0]):
                for j in range(0, data_matrix.shape[1], 9):
                    final_matrix.append(list(data_matrix[i, j:j+9]))
            final_matrix = np.array(final_matrix)
            kmeans = KMeans(n_clusters=400, random_state=0).fit(final_matrix)
            final_store_matrix = np.zeros((data_matrix.shape[0], 400))
            for i in range(data_matrix.shape[0]):
                start = i * data_matrix.shape[1]//9
                print(start)
                for j in range(data_matrix.shape[1]//9):
                    final_store_matrix[i, kmeans.labels_[start+j]] += 1
            data_matrix = final_store_matrix

        comp, proj_mat, feature_vec = nmf(data_matrix, k)

    file_name = test_path + "_" + color_model + "_" + drt + "_" + str(k) + ".csv"

    proj_mat = proj_mat.tolist()
    image_ids = image_ids.tolist()
    cols = [color_model+"_"+drt+"_"+str(i) for i in range(1,k+1)]
    data_frame = pd.DataFrame(data=proj_mat,columns=cols,index=image_ids)
    if os.path.exists(file_name) == True:
        os.remove(file_name)
    ''' Storing the object latent semantics to .csv file'''
    data_frame.to_csv(file_name)

    if (drt == 'LDA' and color_model=='CM') or (drt == 'NMF' and color_model=='CM'):
        return kmeans, comp, feature_vec

    return comp, feature_vec

if __name__ == '__main__':

    test_folderpath=input('Please give the path to images folder')

    color_model = input("Enter the model'CM' for Color Moments\n"
                        "'LBP' for Local Binary patterns:\n"
                        " 'HOG' for Histogram of Gradients:\n"
                        " 'SIFT' for Scale In-variant feature Transform ")

    k = int(input("Enter the value of k to find k-latent semantics: "))

    drt = input("Enter the dimensionality reduction technqiue:\n"
                    "give PCA: Principle Component Analysis\n "
                    "SVD: Singular Value Decompostion\n "
                    "LDA: Latent Dirichlet Allocation\n"
                    "NMF: Non-Negative Matrix Factorization\n")



    # _, co, feature_vec = k_latent_semantics(test_path="testing", color_model="CM", k=15, drt="LDA")

    ''' Calling the method defined above to extract the feature-latent semantics'''

    if (drt == 'LDA' and color_model=='CM') or (drt == 'NMF' and color_model=='CM') :
        _, co, feature_vec = k_latent_semantics(test_path=test_folderpath, color_model=color_model, k=k, drt=drt)
    else:
        co, feature_vec = k_latent_semantics(test_path=test_folderpath, color_model=color_model, k=k, drt=drt)

    term_weight_pairs = {}

    ''' Calculating the term-weight pairs of feature-latent semantic matrix 
    and writing them to .csv file '''

    for i in range(k):
        feature_indexes = np.argsort(feature_vec[i])
        feature_indexes = feature_indexes[::-1]
        pairs = [(j,(feature_vec[i][j])) for j in feature_indexes]
        length = len(pairs)
        term_weight_pairs["latent_semantic" + str(i+1)] = pairs


    term_weight_keys = list(term_weight_pairs.keys())
    term_weight_values = list(term_weight_pairs.values())
    colns = ["term-weight-"+str(i+1) for i in range(length)]
    data_frame1 = pd.DataFrame(data=term_weight_values,columns=colns,index=term_weight_keys)
    term_weight_name = test_folderpath + "_" + color_model + "_" +  drt + "_" + str(k) + "TW.csv"

    if os.path.exists(term_weight_name) == True:
        os.remove(term_weight_name)
    data_frame1.to_csv(term_weight_name)

    print("Open the file " + term_weight_name + " to see the output")











