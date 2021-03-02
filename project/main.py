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
    img = cv2.imread('../img/'+image_name, 0) 
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=7)
    if not os.path.exists('../img/generated/'):
        os.makedirs('../img/generated/')
    cv2.imwrite('../img/generated/temp.jpg', sobel)
    
    img = cv2.imread('../img/generated/temp.jpg', 0) 
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    os.remove('../img/generated/temp.jpg')
    if not os.path.exists('../img/generated/segmentation'):
        os.makedirs('../img/generated/segmentation')
    cv2.imwrite('../img/generated/segmentation/' + str(uuid.uuid1()) + '.jpg', thresh)

    if ifshow:
        show_image(thresh)


def image_analysis():
    parser = argparse.ArgumentParser(description='--Analyse Image--')
    parser.add_argument('--image', 
                        metavar = '<file_name>', 
                        help = 'image file to parse', 
                        required = True)
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