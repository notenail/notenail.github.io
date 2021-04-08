---
layout: post
title: "Introduction to  Segmentation"
description: A brief intro to image segmenation
# image: /files/blog/gibbs/front.jpg
date: 2021-2-10
categories: post
nav-short: true
show-avatar: false
tags: [Blog, Image-Segmentation, Computer Vision]
---


The object detection and classification task make the most buzz in the break-through of machine learning. But one of the most important tasks in computer vision is segmentation. In this blog, I’ll be detailing the application of segmentation-related tasks and in my next blog, I'll introduce you to an instrumental model that revolutionalized the learning-based segmentation methods.

## Segmentation
An image is truly worth a thousand worth. Think about it. A 28x28 image has just 784 data points. However, this image is quite sufficient to express any event that you can think of. That beautiful sunset, the picturesque mountains, the birth of your first child, and so on.  The things that can be expressed with these measly 784 data points in infinite. But as marvelous as the information capacity of an image is, it poses a challenge in image processing. The act of interpretation is inherently human. Computers cannot do this task very well. 

If we want the computers to make some deterministic sense of images, we have to simplify the images. Simplification brings us to segmentation. **Image segmentation refers to the act of creating segments or groping pixels using some meaningful criteria**.     

<!-- [put image here] -->
<!-- img/projects/segmentation/img_seg.jpeg -->
<img src="{{ site.url }}/img/projects/segmentation/img_seg.jpeg" width="100%" hight="100%"> 


Segmentation is primarily of two types.

## Semantic Segmentation
The task of semantic segmentation is to group pixels using a class label, i.e., classification of each pixel to the class it belongs to. For example in the image shown below, all the pixels representing the humans are of the same color(bright pink), and those that represent the automobiles are different(blue). 

<img src="{{ site.url }}/img/projects/segmentation/semantic_seg.jpg" width="100%" hight="100%"> 



## Instance Segmentation
If you merge object detection with semantic segmentation, you get instance segmentation. The only difference between object detection and instance segmentation is that instead of a bounding box around the object, you get a contour or a mask. 

<img src="{{ site.url }}/img/projects/segmentation/instance_seg.png" width="100%" hight="100%"> 

The rectangles around the balloons are essentially what you get if you are using object detection but the masks represented by various colors are the output of a typical instance segmentation model. 

Alright, you now know different types of segmentation. In the next post, we are going to have a detailed look at the architecture of the legendary UNet.  
