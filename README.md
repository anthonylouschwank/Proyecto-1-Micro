# Emotion-Recognition

## Author

[![Linkedin: Thierry Khamphousone](https://img.shields.io/badge/-Thierry_Khamphousone-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/tkhamphousone/)](https://www.linkedin.com/in/tkhamphousone)

Mathis Grandemange

François Lelievre

<br/>

## Getting Started

__Setup__
```bash
> git clone https://github.com/Yulypso/Fruit-Recognition.git
> cd Fruit-Recognition
> python3 -m venv .venv

# for MacOs/Linux
> source .venv/bin/activate

#for Windows
> py -3 -m venv .venv
> .venv\scripts\activate

# to install requirements 
> pip3 install -r requirements.txt
```

__[Check Dependency Graph](https://github.com/Yulypso/Fruit-Recognition/network/dependencies)__

<br/>

__Note__: In Visual Studio Code, don't forget to select the correct Python interpreter. <br/>

[CMD + SHIFT + P] > select Interpreter > Python 3.9.0 64-bits ('.venv') [./.venv/bin/python]

<br/>

**Fruit images**

Download trainset and testset images: [here](https://mega.nz/file/rDQngIAL#Wncxuyb1oBoI8CSDBAE5dCG2WuAP6ViXQXaWPsVH0RU)

Move download images to their respective folder

img/[downloaded images]

Note: **img/** directory must be at the same location as **project/** directory

<br/>

__Run the code__
```bash
> cd project
> python3 main.py -f -c
```

```bash
> python3 main.py [-h] [--fit] [--classify]

--Form detection--

optional arguments:
  -h, --help      show this help message and exit
  --fit, -f       Extract attributes and add them to the classifier
  --classify, -c  Knn algorithm to classify fruits
```

<br/>

## Table of Contents

- __[Introduction](#introduction)__
- __[Thresholding](#thresholding)__
- __[Attribute selection and calculation](#attribute-selection-and-calculation)__
- __[Majority Color](#majority-color)__
- __[Average Color](#average-color)__
- __[Classification](#classification)__
- __[Discussion](#discussion)__

<br/>

## Introduction

The realized project concerns the recognition of forms by computer which is a topic of machine learning or automatic learning which could be useful, in particular in the commercial field where the analysis of images will make it possible to facilitate the detection of the fruits and vegetables in the "intelligent scales" in the supermarkets. 

<br/>

## Thresholding

We chose to use thresholding with parameters that we set after changing color space, ie from RGB to HSV. 
Justification: This segmentation method allows us to recover the shape and texture of the fruit without taking into account the background and shadows unlike a Sobel filter for example. 

<br/>

<p align="center" width="100%">
    <img align="center" width="580" src="https://user-images.githubusercontent.com/59794336/111849759-ad998900-890e-11eb-9b0f-55fa77986968.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 1</strong> - Fruits Thresholding
</p>

<br/>

We notice that the thresholding allowed us to obtain the shape of the fruit while eliminating a large part of the shadow (if existing) of the fruit. 

<br/>

**Strengths and weaknesses of the method :**

**Strengths:** Thresholding is an inexpensive method in terms of memory and time unlike the Sobel filter which requires convolution by applying a horizontal and vertical gradient kernel.

**Weaknesses:** The threshold was defined by hand and could be badly defined for other types of fruits. 


**Areas for improvement:** We could make sure to have an adaptive threshold that is defined according to the different parameters Hue, Saturation, Value or Red Green Blue depending on the color space. 

<br/>

## Attribute selection and calculation

### Zernike moments

The fruits present have a different shape from each other, so the comparison of shapes seems relevant to recognize the fruits. In theory this method should be problematic on fruits like oranges or tomatoes because they have a similar shape. To compare the shape of the fruits we decided to use the Zernike polynomial method.
 
Zernike polynomials are a series of orthogonal polynomials generally used in optics and which can be decomposed as an even or odd function.

<br/>

<p align="center" width="100%">
    <img align="center" width="380" src="https://user-images.githubusercontent.com/59794336/111849977-43351880-890f-11eb-9625-cbd398f69bc0.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 2</strong> - Plots of the Zernike polynomials on the unit disk
</p>

<br/>

In our case we use the Zernike method of moments to obtain from the contour of one of the fruits a descriptor corresponding to the shape. 
The moments allow us to calculate values such as the surface of the object, the center of gravity (the center of the object, in terms of x, y coordinates). The interest of this method is that it does not take into account the rotation of the object, which is an extremely interesting property when we want to work with shape descriptors.

We first perform a segmentation by thresholding on the images to obtain the shape of the fruit in white. Once this step is done, we apply the Zernike method with the image as a parameter, we then obtain a table of 25 numerical values depending on the shape of the object that we will try to reduce to a single value. We then average the 25 values to obtain a 1 feature descriptor.
 
By performing a Knn on the shape descriptor we see that we arrive at an accuracy of 71% with an optimal number of 5 neighbors. 

<br/>

## Majority color

Fruits usually have different colors, so the choice of the attribute "majority color" seems to be relevant. The only fruits in our database that can potentially cause problems are lemons and bananas.

We decided to implement the K-means algorithm where the randomly initialized centroids move to the cluster centers. We then recover the position of centroids that correspond to the majority color written on 3 channels on values between 0 and 255 inclusive. We change the color space from RGB to HSV in order to recover the 'Hue' parameter.

Note: A centroid is associated with a majority color. 

Generally in our database, the white/gray tint is often the majority color and this is why if the value is too close to 255 (which in the HSV encoding corresponds to a white tint as well), we do not define it as the majority color and take the next majority color given by the 2nd centroid of the K-means.

To obtain the majority color of an image, we use a K-means with 4 centroids, which is the optimal number after several experiments. On the other hand, we have seen in a publication that for an image, generally the most optimal number was always less than 5. 

<br/>

## Average color

We have implemented the attribute "average color" which as for the majority color will return a value of the parameter "Hue" in the HSV database of the image. 

- In a first step, we create a mask by performing a segmentation using a thresholding followed by a dilation with a kernel of Elliptic form of size 3x3 to have a black background and the fruit in white.

- In a second step, we create a new image using the '&' operator to obtain a new image with the color of the image and a black background. 

	=> the mask has a black background and given that the value of black is equal to 0, applying the mask on the original image amounts to calculating 0 and pixel_value -> 0 and thus we keep the black color in the background. 

  => Conversely, 1 and pixel_value for the white part of the mask allows us to recover the color of the pixel_value of the original image. 

- Finally we apply the calculation of the average by creating a new mask to not take into account the pixels below 2 (pixel near black). Thus, we obtain an average color that is not altered by the total number of pixels in the image but only by the number of pixels composing the fruit.

<br/>

## Classification 

Selected and implemented method:

We implemented the Knn algorithm in order to perform the classification of the fruits, which looks for the optimal k by itself, and gives us the best result. 



Result obtained and comment:

After choosing the parameters of our algorithm, we can now run it to evaluate its performance:

<br/>

<p align="center" width="100%">
    <img align="center" width="380" src="https://user-images.githubusercontent.com/59794336/111850170-d5d5b780-890f-11eb-889d-a02c2d7d437c.png"/>
</p>

<p align="center" width="100%">
    <strong>Figure 3</strong> - Classification result
</p>

<br/>

## Discussion

The images of the fruits provided are realistic because shadows can be observed in some images and the fruits are recognizable, however they are not representative of the situation mentioned in the introduction because most of the images we have in the database have a single color white or close to white background. (Value of the pixel = 255) 


A realistic image would be a picture taken directly from the bottom of the supermarket's fruit scale and would differ by : 

- The number of fruits on the same image will impact the thresholding by creating shadows on other fruits
- The Zernike moment will not be used anymore because the number of fruits will impact the final shape of the fruit after thresholding and post-processing. 
- The fruit in the supermarket is usually enclosed in a relatively opaque plastic bag which could have a big influence on the shape detection. 
- Poor lighting on the scale pan or dirt on the camera could also impact the photo quality and thus indirectly our classification model. 
- On a scale, the images of the fruits will keep their proportionality and therefore will be located at equal distance from the camera. It will therefore be easier for the algorithm to have the attribute "fruit size" in addition in its features before training the model. 


We will have to readjust our program, especially in terms of segmentation and post-processing, to deal with plastic bags, the number of fruits, bad lighting and other hazards. We will also be able to add other attributes such as the size and weight of the fruit (recoverable from the scale). 
Our classification model can then be optimized with a more complete dataset with fruit measurement attributes.

