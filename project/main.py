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


def show_image(img, file):
    cv2.imshow(file, img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()     


def clear_generated_dir():
    shutil.rmtree('../img/generated')


def auto_Canny(img, sigma=0.33):
    v = np.median(img)
    low = int(max(0, 1.0 - sigma) * v)
    up = int(min(255, 1.0 + sigma) * v)
    return cv2.Canny(img, low, up)


def otsu_canny(image, lowrate=0.1):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)
    return edged


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
    for file in sorted(glob.glob(os.path.join(path, '*.*'))):
        label.append(str(file.split('/')[2].split('_')[0]))
    return label


def f_knn(data, label):
    '''
    knn algorithm 
    '''
    # searching for the best k
    knn_scores = []
    knn_classifiers = []
    for i in range(1, 12):
        knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute', n_jobs=-1)
        knn_scores.append(cross_val_score(knn, data, label, cv=7, n_jobs=-1, scoring='accuracy').mean())
        knn_classifiers.append(knn)
    knn_index = knn_scores.index(np.max(knn_scores))
    best_k = knn_scores[knn_index]
    print(f'Best k is {knn_index+1} for {np.round(knn_scores[knn_index], 2)} accuracy')

    # training with the best k then predicting
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.7, random_state=46)
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
    plt.title('Taux de reconnaissance en fonction du nombre de voisins K')
    plt.xlabel('Nombre de voisins K')
    plt.ylabel('Taux de reconnaissance')
    plt.plot(range(1, 12), knn_scores)
    plt.axhline(knn_scores[knn_index], color='r')
    plt.axvline(knn_index+1, color='r')
    plt.show()


def f_kmeans():
    pass


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
        #img2 = img2&img
        img[img >= 255] = 0
        red = np.mean(img[:,:,0]) 
        green = np.mean(img[:,:,1]) 
        blue = np.mean(img[:,:,2]) 
        img[:,:,0] = red
        img[:,:,1] = green
        img[:,:,2] = blue
        img_out = img2 & img
        show_image(img_out, file)


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
    args = parser.parse_args()
    clear = args.clear
    segmentation1 = args.segmentation1
    segmentation2 = args.segmentation2
    segmentation3 = args.segmentation3
    knn = args.knn
    kmeans = args.kmeans

    if clear:
        clear_generated_dir()

    path = '../img/'

    '''
    1. Segmentation
    '''
    if segmentation1:
        f_segmentation1(path) #TODO : retourner un tableau elongation ? 
    
    elif segmentation2:
        texture = f_segmentation2(path) #TODO: retourner un tableau compacite 
    
    elif segmentation3:
        f_segmentation3(path) # TODO: retourner un tableau couleur_dominante
    
    '''
    2. Calcul des attributs
        - Couleur dominante (avocat/orange)
        - Compacité (fraise/tomate)
        - Elongation (banane/citron)

    (placer l’ensemble des attributs calculés pour chaque objet dans un vecteur 'data')
    '''
    label = f_get_label(path) #vecteur contenant dans le meme ordre les labels des fruits
    print('taille echantillon:', len(label))

    
    # TODO
    data = [] # vecteur data qu'on utilisera pour nos modèles
    for i in range(len(label)):
        data_row = [] # 1 ligne = 1 echantillon = n features 
        data_row.append(texture[i].flatten())
        #data_row.append(compacite[i].flatten()) #TODO
        #data_row.append(elongation[i].flatten()) #TODO
        #data_row.append(couleur_dominante[i].flatten()) #TODO
        data.append(data_row[0]) # on ajoute toutes les features de l'échantillon i
    data = np.array(data)
    print(np.shape(data))

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