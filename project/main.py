#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, uuid, cv2, argparse, shutil, glob, colorsys, mahotas, time, operator, functools, platform
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

# unused
def clear_generated_dir():
    shutil.rmtree('../img/generated')

# Unused
def transform(nested_list):
    regular_list=[]
    for ele in nested_list:
        if type(ele) is list:
            regular_list.append(ele)
        else:
            regular_list.append([ele])
    return regular_list

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

# unused
def f_zernike_reduced(tab):
    tab_reduced=[]
    for i in tab:
        tab_reduced.append(np.mean(i))
    return tab_reduced    

# unused
def f_count(path):
    '''
    1. get textures through canny filter
    2. Count the nb of white pixels
    '''
    counts = []
    for i, file in enumerate(sorted(glob.glob(os.path.join(path, '*.*')))):
        print("Processing count: "+str(i+1)+" on 54", end='\r')
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
        ret[ret > 10] = 255
        ret[ret <= 10] = 0
        colors, counter = np.unique(ret, return_counts=True, axis=None)
        counts.append(counter[1])
    print('')
    return counts


class ZernikeMoments:
    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius)


def get_elongation(m):
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
    return (x + y**0.5) / (x - y**0.5)


def show_image(img, file):
    cv2.imshow(file, img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()     


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
    print(f'\nBest k is {knn_best_k[knn_index]} for {np.round(knn_scores[knn_index], 2)} accuracy')
    #print('params:', knn_classifiers[knn_index].get_params)

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
    plt.scatter(range(1, len(knn_scores)+1), knn_scores)
    plt.axhline(knn_scores[knn_index], color='r')
    plt.axvline(knn_index+1, color='r')
    plt.show()


def f_majority_color(path):
    couleurs=[]
    for i, file in enumerate(sorted(glob.glob(os.path.join(path, '*.*')))):
        print("Processing K-means for majority color: "+str(i+1)+" on 54", end='\r')
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
        tmp=Counter(clt.labels_)        
        chosen=tmp.most_common(1)[0]
        chosen=clt.cluster_centers_[chosen[0]]
        chosen=colorsys.rgb_to_hsv(chosen[0],chosen[1],chosen[2])
        if chosen[2]>200 :
            chosen=tmp.most_common(2)[1]
            chosen=clt.cluster_centers_[chosen[0]]
            chosen=colorsys.rgb_to_hsv(chosen[0],chosen[1],chosen[2])
        couleurs.append(chosen[0])
    print('')
    return couleurs
    

def f_zernikes(path):
    '''
    Zernikes moments
    '''
    desc = ZernikeMoments(100)
    tab = []
    for i, file in enumerate(sorted(glob.glob(os.path.join(path, '*.*')))):
        print("Processing Zernikes moments: "+str(i+1)+" on 54", end='\r')
        img = cv2.imread('../img/'+file) 
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([1,60,50])
        upper = np.array([255,255,255])
        result = cv2.inRange(result, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        result = cv2.dilate(result,kernel)
        moments = desc.describe(result)
        tab.append(np.mean(moments))
    print('')
    return tab


def f_elongation(path):
    '''
    1. Threshold: fruit forms in white color and background in black color
    2. get elongation attribute
    '''
    elongations = []
    for i,file in enumerate(sorted(glob.glob(os.path.join(path, '*.*')))):
        print("Processing elongation: "+str(i+1)+" on 54", end='\r')
        img = cv2.imread('../img/'+file) 
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([1,45,0])
        upper = np.array([255,255,255])
        result = cv2.inRange(result, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        result = cv2.dilate(result,kernel)
        resized = cv2.resize(result, (300,250))
        resized[resized > 0] = 255
        m = cv2.moments(resized)
        elongations.append(get_elongation(m))
    print('')
    return elongations
        

def f_mean_color(path):
    '''
    Mean color
    '''
    mean_colors = []
    for i, file in enumerate(sorted(glob.glob(os.path.join(path, '*.*')))):
        print("Processing mean color: "+str(i+1)+" on 54", end='\r')
        img = cv2.imread('../img/'+file)  
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([1,60,50])
        upper = np.array([255,255,255])
        result = cv2.inRange(result, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        result = cv2.dilate(result, kernel)
        new_img = np.zeros_like(img)
        new_img[:,:,0]=result
        new_img[:,:,1]=result
        new_img[:,:,2]=result
        img2 = img & new_img
        mask_r = np.ma.masked_less(img2[:,:,0],2)
        mask_g = np.ma.masked_less(img2[:,:,1],2)
        mask_b = np.ma.masked_less(img2[:,:,2],2)
        red = np.mean(mask_r) 
        green = np.mean(mask_g) 
        blue = np.mean(mask_b) 
        new_img[:,:,0] = red
        new_img[:,:,1] = green
        new_img[:,:,2] = blue
        hsv_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        mean_colors.append(h[0][0])
    print('')
    return mean_colors


def image_analysis():
    parser = argparse.ArgumentParser(description='--Form Detection--')
    parser.add_argument('--fit',
                        '-f',
                        help = 'Extract attributes and add them to the classifier',
                        action = 'store_true',
                        required = '--classify' in sys.argv or '-c' in sys.argv
                        )
    parser.add_argument('--classify',
                        '-c',
                        help = 'Knn algorithm to classify fruits',
                        action = 'store_true',
                        required = False
                        )
    args = parser.parse_args()
    fit = args.fit
    classify = args.classify

    path = '../img/'
    majority_color_res, zernike_moments_res, mean_color_res, count_res, elongation_res = None, None, None, None, None
    '''
    1. Segmentation 
    2. Post-processing
    3. Attribute calculation
    '''
    if fit:
        elongation_res = f_elongation(path)
        mean_color_res = f_mean_color(path) 
        majority_color_res = f_majority_color(path)
        zernike_moments_res = f_zernikes(path)
        label = f_get_label(path) # vecteur contenant dans le meme ordre les labels des fruits

        print('\nSample size:', len(label), 'labels:', np.unique(label))
        print('Elongation shape', np.shape(elongation_res))
        print('Zernike moments shape', np.shape(zernike_moments_res))
        print('Majority color shape', np.shape(majority_color_res))
        print('Mean color shape', np.shape(mean_color_res))

        print(count_res)

        data = [] # vecteur data qu'on utilisera pour nos modèles
        for i in range(len(label)):
            feature_list = [] # liste des features à ajouter à la data_row
            feature_list.append(majority_color_res[i]) 
            feature_list.append(elongation_res[i])
            feature_list.append(mean_color_res[i])
            feature_list.append(zernike_moments_res[i])
            data_row = np.concatenate(feature_list, axis = None)
            data.append(data_row) # on ajoute toutes les features de l'échantillon i
        
        data = np.array(data)
        print('data shape:', np.shape(data), 'nb_rows:', np.shape(data)[0], 'nb_columns:', np.shape(data)[1])
    
    '''
    4. Classification
    '''
    if classify:
        f_knn(data, label)


def main():
    image_analysis() 
    sys.exit(0)


if __name__ == "__main__":
    main()