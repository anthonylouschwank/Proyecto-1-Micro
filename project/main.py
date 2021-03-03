#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, uuid, cv2, argparse, shutil
import numpy as np


def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()     


def clear_generated_dir():
    shutil.rmtree('../img/generated')


def f_segmentation(image_name, ifshow):
    
    img = cv2.imread('../img/'+image_name) 
    if ifshow:
        show_image(img)

    img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ifshow:
        show_image(img)

    # pour retirer le bruit
    low_dilated_img = cv2.dilate(img_bin, np.ones((3,3), np.uint8))
    if ifshow:
        show_image(low_dilated_img)

    dilated_img = cv2.dilate(img_bin, np.ones((50,50), np.uint8))
    if ifshow:
        show_image(dilated_img)

    # pour atténuer le bruit de l'image dilaté
    bg_img = cv2.medianBlur(dilated_img, 7)
    if ifshow:
        show_image(bg_img)

    diff_img = 255 - cv2.absdiff(low_dilated_img, bg_img)
    if ifshow:
        show_image(diff_img)

    _, img_th = cv2.threshold(diff_img, 220, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    if ifshow:
        show_image(img_th)

    img_floodfill = img_th.copy()
    h, w = img_th.shape[:]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0,0), 255);

    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    if ifshow:
        show_image(img_floodfill_inv)



    img_out = img_th | img_floodfill_inv 
    if ifshow:
        show_image(img_out)

    # convert into 3 dimensions
    img2 = np.zeros_like(img)
    img2[:,:,0] = img_out
    img2[:,:,1] = img_out
    img2[:,:,2] = img_out

    # convert white background to black
    img[img >= 250] = 0

    # get mean color channel
    green_channel = np.mean(img[:,:,0])
    blue_channel = np.mean(img[:,:,1])
    red_channel = np.mean(img[:,:,2])
    img[:,:,0] = green_channel
    img[:,:,1] = blue_channel
    img[:,:,2] = red_channel

    # fill floodfilled image with mean color
    img_out = img2 & img

    show_image(img_out)
    if ifshow:
        show_image(img_out)

    if not os.path.exists('../img/generated/'):
        os.makedirs('../img/generated/')
    if not os.path.exists('../img/generated/segmentation'):
        os.makedirs('../img/generated/segmentation')
    cv2.imwrite('../img/generated/segmentation/' + str(uuid.uuid1()) + '.jpg', img_out)

    #Sobel edge detection
    #img = cv2.GaussianBlur(img,(5,5),0)
    #ret, otsu = cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #show_image(otsu)

    #sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
    #if not os.path.exists('../img/generated/'):
    #    os.makedirs('../img/generated/')
    #cv2.imwrite('../img/generated/temp.jpg', sobel)
    #show_image(sobel)
    

def image_analysis():
    parser = argparse.ArgumentParser(description='--Analyse Image--')
    parser.add_argument('--image', 
                        metavar = '<file_name>', 
                        help = 'image file to parse', 
                        required = False)
    parser.add_argument('--segmentation',
                        '-s',
                        help = 'segmentation',
                        action = 'store_true',
                        required = False)
    parser.add_argument('--show',
                        '-sh',
                        help = 'show image',
                        action = 'store_true',
                        required = False)
    parser.add_argument('--clear',
                        '-c',
                        help = 'clear generated images',
                        action = 'store_true',
                        required = False)
    args = parser.parse_args()
    image_name = args.image
    segmentation = args.segmentation
    show = args.show
    clear = args.clear

    if image_name:
        print('--- Analyse d\'image ---')
        print("Travail de l'image: " + image_name)

    if clear:
        clear_generated_dir()

    if segmentation:
        f_segmentation(image_name, show)
        

def main():
    image_analysis() 
    sys.exit(0)


if __name__ == "__main__":
    main()