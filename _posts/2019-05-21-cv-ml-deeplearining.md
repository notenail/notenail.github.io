---
layout: post
title: "Deep Learning vs Computer Vision"
description: My thoughts on the debate of computer vision vs deep learning.
# image: /files/blog/gibbs/front.jpg
date: 2019-05-21
categories: post
tags: [Blog, Computer Vision, Machine Learning, Deep Learning]
---

I am compelled to write this post due to the various trends that I observed and the presence of misinformation related to computer vision and deep learning. This post is aimed at professionals and students who are developing expertise in deep learning and are keen to start a career in computer vision. 

First, about me. I work as a Data Scientist in a start-up whose domain is Agricultural technology. We are developing algorithms that, simply put, can analyze the pictures such as below and analyze the product on various parameters such as shape, size, color, sugar contents, etc. Our data is primarily based on images and videos containing the concerning crop/product. I am writing this post to help and at many points it might feel like a rant against deep learning, however, if you make it all the way though, you’d get to know what I mean by the following headline.

<img src="{{ site.url }}/img/objects.png" width="100%">

## Computer Vision is not Deep Learning 

I had to reject a ton of candidates who applied some deep learning off the shelf algorithms such as yolo or mobilenet on images and videos. I have fairly recent and the relevant experience of computer vision and I can tell you that deep learning is not computer vision. You can create a pipeline of an ML algorithm without knowing the basic principles and algorithms in computer vision and it will work wonderfully. But that is exactly what you would have created. How would you go about processing the inferences? And even more relevant, why did you create that pipeline in the first place? Was it even necessary to introduce a deep learning overhead in an application such as object detection? I know what you are thinking. Object detection and deep learning goes hand in hand. The statement is quite alright but deep learning is not the only way to perform such task and more often than not, it’s not necessary to use a deep learning algorithm.

<img src="{{ site.url }}/img/gpu.png" width="100%">

In the real world of computer vision, you’d be surprised to know that most of the applications that involve object detection can be handled by a good computer vision library containing old school algorithms like Haar-feature detection or Histogram of gradients . You don’t even need to introduce deep learning in cases where say you have a belt on which a certain type of object has to be detected.

To be fair, deep learning do facilitates operations such as object detection and segmentation in a complex challenging data distribution. There are many problems in computer vision that the conventional algorithms have struggled but the newer CNN based algorithms, given sufficient data, have provided us with giant break-through.

Being said that, you would still need to know about the color spaces, exposure, conventional feature detector such as SHIFT and SURF, contouring, block filtering, denoising, transformations, etc, just to know how to start processing a digital image. And that’s usually the very beginning of any project.

These are mostly statistical and deterministic tools. Knowledge of statistics is a must in any data-driven field, be it computer vision or NLP. I’ve to constantly study statistical modelling because in practice they have as much power as the latest deep learning techniques and that too without being a black-box in nature. They take a very low amount of data to model and an even lower amount of hardware to deploy as compared to deep learning counterparts. The thing with statistical modelling is that the modeller has to be smart enough to not only know “what” to look for but also “where”, the “how” is choice that a data scientist has to make based on available resources.


Sometimes we have a plethora of data and if the application is simple enough, we use machine learning techniques including deep learning along with the aforementioned statistical techniques. As opposed to this, if the data is scarce, which is usually the case with new businesses, we use tools like data-visualization to understand the problem better and then apply an algorithm of an educated choice. A solid understanding of data and features gives any data scientists an extra acceleration while performing critical decision making using an EDA.


Nevertheless, I have to tell you that deep learning is an essential tool for any data scientist today, irrespective of the field that they are working. But here’s a thing about deep learning,

## Deep Learning is not only the course you took online!

The first thing that I notice while hiring for the role of a data scientist or computer vision is the projects that the candidates have done. Usually, an applicant’s CV is full of courses offered by the University of the internet. That is actually quite fine as I personally would have not been in this particular field if it wasn’t for the internet. However, once a candidate has finished a certificate course, they start applying for companies as a data scientist. This should not be the case. What potential employers would like to see is a solid project other than that of the online course. We are fortunate enough to be living in an extremely data-rich era. Find one that interests you and try to make inferences based on the knowledge you have.

Most of the candidates I have screened don’t even have a kaggle profile. I visited kaggle 17 times last week alone. Kaggle is the real place of learning for me. There are experts teaching you the best available techniques for a variety of datasets.  Having said that, time for a yet another bitter truth(IMO).

## Deep Learning doesn't encompassess Machine Learning

Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to effectively perform a specific task without using explicit instructions, relying on patterns and inference instead. Please understand this statement.

<img src="{{ site.url }}/img/blog1.png" width="100%">


Deep learning is an over-simplified word which is used to categorise the family of neural networks with tens of hidden layers more as compared to conventional ANNs. That’s all it is. It is NOT equivalent to machine learning. In reality, it is a small subset of machine learning. A data scientist with the sole experience of deep learning is not the perfect data scientist. In fact, in our hiring process, that person won’t even qualify as a data scientist. There is also a great deal of misinformation online about machines thinking and making decisions as humans do, which is incredibly misleading and ultimately untrue.

Finally, if you want to start your career as a data scientist working in computer vision, master the following libraries/courses or something that might substitute them, preferably in python and C++.

- Image Processing: OpenCV, skimage
- Data Analysis and Machine Learning: Sklearn
- Data Visualisation: Matplotlib
- Data Manipulation: Numpy
- Statistics: Pandas, scipy
- Deep Learning: Keras

To learn introductory image processing you can start this mini tutorial series by wonderful sentdex. For advance learners, I would recommend this course, it is really good and tbh, quite interesting. Good Luck!