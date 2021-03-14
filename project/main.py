#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, uuid, cv2, argparse, shutil, glob
from matplotlib import pyplot as plt
import numpy as np


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


def f_segmentation1(path):
    for file in glob.glob(os.path.join(path, '*')):
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
    for file in glob.glob(os.path.join(path, '*')):
        img = cv2.imread('../img/'+file) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.GaussianBlur(img, (5,5), 2)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8) 
        erosion = cv2.erode(res, kernel)
        dilation = cv2.dilate(erosion, kernel)
        binaryimg = cv2.Canny(res, 50, 200) 
        h = cv2.findContours(binaryimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
        contours = h[0]
        ret = np.ones(res.shape, np.uint8) 
        cv2.drawContours(ret,contours,-1,(255,255,255),1) 
        show_image(ret, file)


def f_segmentation3(path):
    for file in glob.glob(os.path.join(path, '*')):
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

        '''
        if not os.path.exists('../img/generated/'):
            os.makedirs('../img/generated/')
        if not os.path.exists('../img/generated/segmentation'):
            os.makedirs('../img/generated/segmentation')
        cv2.imwrite('../img/generated/segmentation/' + str(uuid.uuid1()) + '.jpg', img2)
        '''


def f_sandbox(path):
    for file in glob.glob(os.path.join(path, '*')):
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
    parser.add_argument('--clear',
                        '-c',
                        help = 'clear generated images',
                        action = 'store_true',
                        required = False)
    args = parser.parse_args()
    segmentation1 = args.segmentation1
    segmentation2 = args.segmentation2
    segmentation3 = args.segmentation3
    clear = args.clear

    if clear:
        clear_generated_dir()

    path = '../img/'

    if segmentation1:
        f_segmentation1(path)
    
    elif segmentation2:
        f_segmentation2(path)
    
    elif segmentation3:
        f_segmentation3(path)
        

def main():
    image_analysis() 
    sys.exit(0)


if __name__ == "__main__":
    main()