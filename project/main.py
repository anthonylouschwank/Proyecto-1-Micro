#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, uuid, cv2, argparse, shutil, glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import platform
import functools
import operator
import time


def show_image(img, file):
    cv2.imshow(file, img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()     


def clear_generated_dir():
    shutil.rmtree('../img/generated')

# unused
def auto_Canny(img, sigma=0.33):
    v = np.median(img)
    low = int(max(0, 1.0 - sigma) * v)
    up = int(min(255, 1.0 + sigma) * v)
    return cv2.Canny(img, low, up)

# unused
def otsu_canny(image, lowrate=0.1):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)
    return edged

# unused
def f_sandbox(path):
    for file in sorted(glob.glob(os.path.join(path, '*.*'))):
        img = cv2.imread('../img/'+file) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.GaussianBlur(img, (3,3), 0)
        #img = auto_Canny(img)
        img = otsu_canny(img)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        #kernel = np.ones((4,4),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print(np.shape(contours))
        #print(hierarchy)
        #mx = (0,0,0,0)      # biggest bounding box so far
        #mx_area = 0
        #for cont in contours:
        #    x,y,w,h = cv2.boundingRect(cont)
        #    area = w*h
        #    if area > mx_area:
        #        mx = x,y,w,h
        #        mx_area = area
        #x,y,w,h = mx
        #img=img[y:y+h,x:x+w]

        #contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, contourIdx=-1, color=(255, 255, 255),thickness=-1)
        show_image(img, file)
        '''
        if not os.path.exists('../img/generated/'):
            os.makedirs('../img/generated/')
        if not os.path.exists('../img/generated/segmentation'):
            os.makedirs('../img/generated/segmentation')
        cv2.imwrite('../img/generated/segmentation/' + str(uuid.uuid1()) + '.jpg', img2)
        '''


def f_get_label(path):
    label = []
    p = platform.system()
    for file in sorted(glob.glob(os.path.join(path, '*.*'))):
        if p == 'Darwin' or p == 'Linux':
            label.append(str(file.split('/')[2].split('_')[0]))
        elif p == 'Windows':
            label.append(str(file.split('\\')[1].split('_')[0]))
    return label


def f_knn(data, label):
    '''
    knn algorithm 
    '''
    # searching for the best k
    knn_scores = []
    knn_classifiers = []
    knn_random_states = []
    knn_best_k = []
    for i in range(1, 12): # for k neighbors
        for j in range(1, 50): # for random_state
            knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute')
            X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.7, random_state=j)
            knn.fit(X_train, y_train)
            knn_scores.append(knn.score(X_test, y_test))
            knn_classifiers.append(knn)
            knn_random_states.append(j)
            knn_best_k.append(i)
    knn_index = knn_scores.index(np.max(knn_scores))
    print(f'Best k is {knn_best_k[knn_index]} for {np.round(knn_scores[knn_index], 2)} accuracy')
    print('best random states:', knn_random_states[knn_index])
    print('params:', knn_classifiers[knn_index].get_params)

    # training with the best k then predicting
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.7, random_state=knn_random_states[knn_index])
    knn_classifiers[knn_index].fit(X_train, y_train)
    y_pred = knn_classifiers[knn_index].predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    unique_label = np.unique([y_test, y_pred])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=unique_label), 
        index=['actual:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    print(cmtx)

    # display graph
    plt.title('Taux de reconnaissance en fonction du nombre de voisins K et random_state')
    plt.xlabel('Nb of neighbors K[1,12], random_state[1,50]')
    plt.ylabel('accuracy')
    plt.plot(range(1, len(knn_scores)+1), knn_scores)
    plt.axhline(knn_scores[knn_index], color='r')
    plt.axvline(knn_index+1, color='r')
    plt.show()


def f_kmeans():
    pass


def couleurmaj(path):
    couleurs=[]

    for i, file in enumerate(sorted(glob.glob(os.path.join(path, '*.*')))):
        print("Processing K-means for majority color: "+str(i)+" on 54", end='\r')
        img = cv2.imread('../img/'+file) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        clt = KMeans(n_clusters=5, init='k-means++', random_state=3, algorithm='full') 
        clt.fit(img)

        #numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        #(hist, _) = np.histogram(clt.labels_, bins = numLabels)
        #hist = hist.astype("float")
        #hist /= hist.sum()
        #bar = plot_colors(hist, clt.cluster_centers_)
        #plt.figure()
        #plt.axis("off")
        #plt.imshow(bar)
        #plt.show()
        couleurs.append(list(clt.cluster_centers_.flatten()))
    print('\n')
    return couleurs
    
# unused
def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype = "uint8")   
    startX = 0
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
        color.astype("uint8").tolist(), -1)
        startX = endX
    return bar


def f_segmentation1(path):
    for file in sorted(glob.glob(os.path.join(path, '*.*'))):
        img = cv2.imread('../img/'+file) 
        
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([1,60,50])
        upper = np.array([255,255,255])
        result = cv2.inRange(result, lower, upper)

        #show_image(result, file)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        result = cv2.dilate(result,kernel)
        show_image(result, file)
        

def f_segmentation2(path):
    texture = []
    for file in sorted(glob.glob(os.path.join(path, '*.*'))):
        img = cv2.imread('../img/'+file) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.GaussianBlur(img, (5,5), 2)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8) 
        img = cv2.erode(res, kernel)
        img = cv2.dilate(img, kernel)
        img = cv2.Canny(res, 50, 200) 
        h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours = h[0]
        ret = np.ones(res.shape, np.uint8) 
        cv2.drawContours(ret,contours,-1,(255,255,255),1) 
        resized = cv2.resize(ret, (300,250))
        texture.append(resized)
        #show_image(resized, file)
    return texture


def f_segmentation3(path):
    for file in sorted(glob.glob(os.path.join(path, '*.*'))):
        img = cv2.imread('../img/'+file)  
        img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        low_dilated_img = cv2.dilate(img_bin, np.ones((3,3), np.uint8))
        dilated_img = cv2.dilate(img_bin, np.ones((1000,1000), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 7)
        diff_img = 255 - cv2.absdiff(low_dilated_img, bg_img)
        _, img_th = cv2.threshold(diff_img, 220, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        img_floodfill = img_th.copy()
        h, w = img_th.shape[:]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(img_floodfill, mask, (0,0), 255);
        img_floodfill_inv = cv2.bitwise_not(img_floodfill)
        img_out = img_th | img_floodfill_inv 
        img2 = np.zeros_like(img)
        img2[:,:,0] = img_out
        img2[:,:,1] = img_out
        img2[:,:,2] = img_out
        img[img >= 255] = 0
        mask_r = np.ma.masked_less(img[:,:,0],1)
        mask_g = np.ma.masked_less(img[:,:,1],1)
        mask_b = np.ma.masked_less(img[:,:,2],1)
        red = np.mean(mask_r) 
        green = np.mean(mask_g) 
        blue = np.mean(mask_b) 
        img[:,:,0] = red
        img[:,:,1] = green
        img[:,:,2] = blue
        img_out = img2 & img
        show_image(img_out, file)


# Unused
def transform(nested_list):
    regular_list=[]
    for ele in nested_list:
        if type(ele) is list:
            regular_list.append(ele)
        else:
            regular_list.append([ele])
    return regular_list


def image_analysis():
    parser = argparse.ArgumentParser(description='--Analyse Image--')
    parser.add_argument('--segmentation1',
                        '-s1',
                        help = 'segmentation',
                        action = 'store_true',
                        required = False)
    parser.add_argument('--segmentation2',
                        '-s2',
                        help = 'segmentation',
                        action = 'store_true',
                        required = False)
    parser.add_argument('--segmentation3',
                        '-s3',
                        help = 'segmentation',
                        action = 'store_true',
                        required = False)
    parser.add_argument('--knn',
                        '-kn',
                        help = 'Knn',
                        action = 'store_true',
                        required = False)
    parser.add_argument('--kmeans',
                        '-km',
                        help = 'Kmeans',
                        action = 'store_true',
                        required = False)
    parser.add_argument('--clear',
                        '-c',
                        help = 'clear generated images',
                        action = 'store_true',
                        required = False)
    parser.add_argument('--colormaj',
                        '-cm',
                        help = 'colormaj',
                        action = 'store_true',
                        required = False)
    args = parser.parse_args()
    clear = args.clear
    segmentation1 = args.segmentation1
    segmentation2 = args.segmentation2
    segmentation3 = args.segmentation3
    colormaj=args.colormaj
    knn = args.knn
    kmeans = args.kmeans

    if clear:
        clear_generated_dir()

    path = '../img/'
    texture, compacite, elongation, couleurmajT = None, None, None, None
    '''
    1. Segmentation
    '''
    if segmentation1:
        f_segmentation1(path) #TODO : retourner un tableau elongation ? 
    
    elif segmentation2:
        texture = f_segmentation2(path) #TODO: retourner un tableau compacite 
    
    elif segmentation3:
        f_segmentation3(path) # TODO: retourner un tableau couleur_dominante
    
    if colormaj :
        couleurmajT = couleurmaj(path)

    '''
    2. Calcul des attributs
        - Couleur dominante (avocat/orange)
        - Compacité (fraise/tomate)
        - Elongation (banane/citron)

    (placer l’ensemble des attributs calculés pour chaque objet dans un vecteur 'data')
    '''
    label = f_get_label(path) #vecteur contenant dans le meme ordre les labels des fruits
    print('Sample size:', len(label))
   
    data = [] # vecteur data qu'on utilisera pour nos modèles
    for i in range(len(label)):
        feature_list = [] # liste des features à ajouter à la data_row
        if texture: 
            feature_list.append(texture[i]) # 1.Remove ? 2.get better metrics ? 
        if compacite:
            pass
            # feature_list.append(compacite[i]) #TODO
        if elongation:
            pass
            # feature_list.append(longation[i]) #TODO
        if colormaj:
            feature_list.append(couleurmajT[i]) # DONE 
        if texture or compacite or elongation or colormaj:
            data_row = np.concatenate(feature_list, axis = None)
            data.append(data_row) # on ajoute toutes les features de l'échantillon i

    data = np.array(data)
    print(np.shape(data), 'data')
    
    '''
    3. Classification
    '''
    if knn:
        f_knn(data, label)
    if kmeans:
        f_kmeans()
    # TODO

def main():
    image_analysis() 
    sys.exit(0)


if __name__ == "__main__":
    main()