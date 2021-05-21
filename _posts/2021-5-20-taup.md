---
layout: post
title: "Task Agnostic Unsupervised Pretraining"
description: A brief intro to unsupervised pretraining
# image: /files/blog/gibbs/front.jpg
date: 2021-5-20
categories: post
nav-short: true
show-avatar: false
tags: [Blog, Unsupervised-Learning, Computer Vision]
---

## Should you annotate your data? 

As an engineer whose work concerns learning-based architecture in the field of robotics and automation, data generation is one of the easier challenges to overcome. But it is one thing to generate data in heaps and piles and a completely different thing to get a workable machine learning model out of it. 

Now, if you choose to develop your models using supervised learning, you’ll have to face the daunting task of completely annotating the data. Data annotation or data labeling is one of the primary consumers of capital, energy, and most importantly, time. Hence, supervised learning is the last thing I consider while designing a solution as the CTO of a startup in the times of COVID. 

So what are the alternatives? 
The answer to this question includes semi-supervised, self-learning, and meta-learning. In this blog, we will look at the technique called Task Agnostic Unsupervised Pretraining. 


## Task Agnostic Unsupervised Pretraining

Let’s deconstruct the terms used above. When we want to train a deep neural network, the first thing that we do is to create architecture. Once we have defined our layers, hyperparameters, and activation function, we initialize the neural network weights and start the training epochs. As opposed to the random initialization of the weights, Pretraining aims to set the weights optimally to give the network a headstart. Think about pretraining as the act of obtaining optimal weights at the network initialization so that the model converges in fewer epochs. 

Task agnostic indicates that the pretraining will not be affected by the nature of the task at the hand. It’s more ubiquitous than one might think. One example of Task-agnostic activity is data augmentation in computer vision. As we explore the technique further, it’ll be clear why and how exactly our technique is task agnostic. 

The fact that we need to know right now is that we will be doing all this in an unsupervised fashion. Technically the pretraining will be unsupervised while the learned representation can leverage very few samples of supervised data for finetuning. The subsequent finetuning requires as little as 5% of labeled data than the data requirement of a fully supervised model with the same architecture.  Remarkably, the performance of unsupervised pretraining coupled with supervised finetuning on this tiny labeled subset of data is similar and at times even better than training the model using supervised training, employing far more labeled samples.

A sample repository with TAUP implemented can be located here <a href="https://github.com/nikheelpandey/TAUP-PyTorch" target="_blank" rel="noopener noreferrer" style="color:blue"> at github.</a>


While it would be a tedious task to go into too many details, I try to explain in an abstract manner.

## Model 
Okay, here’s the trick. You use a base learner like ResNet, remove the last fully connected layer (as done in transfer learning), and fit a projection head over this decapitated model by adding a couple of fully connected layers. The model would take the input and would return a 256 dimension vector. After training this model contrastively (<a href="https://nikhilpandey.in/2021-01-20-simCLR/" target="_blank" rel="noopener noreferrer" style="color:blue"> see here.</a>), we will remove the projection head and add a linear layer for classification. 

## Dataset

You can use any dataset as long as you have few labeled samples for the supervised fine-tuning part. The best thing about this approach is that you can use as much data as you can possibly produce because the training is unsupervised. 
As for data augmentation, random-crop, random-horizontal flip, and color jitter should be used. In my testing, these three augmentation techniques yielded a better result than any other combination. 



## Algorithm
The algorithm to train a TAUP model can be broken down into two board parts as I alluded to earlier. Some of the prominent papers from this domain are simCLR, BYOL, SimSiam, etc. They more or less follow the same pattern.

<img src="{{ site.url }}/img/projects/taup/taup.jpg" width="100%" hight="100%"> 

- Take an image and create a pair by augmenting it.
- The goal is to train a model that produces a similar representation of positive pairs and dissimilar representation of negative pairs.


Following algorithm is taken from the BYOL paper.

<img src="{{ site.url }}/img/projects/taup/byol.jpg" width="100%" hight="100%"> 


We force(train) the model to produce a similar representation of positive pairs, i.e, an image and its augmented version. And that’s the crux of the technique. To do this, we use cosine similarity. simCLR and BOYL have a different approach to the use of similarity. simCLR uses positive and negative pairs in the loss function while BOYL just uses positive pairs. Please refer to <a href="https://github.com/nikheelpandey/TAUP-PyTorch" target="_blank" rel="noopener noreferrer" style="color:blue"> simCLR implementation</a> and <a href="https://github.com/nikheelpandey/BYOL-PyTorch" target="_blank" rel="noopener noreferrer" style="color:blue"> this</a> for a BOYL implementation. 

Once we train the base learner with the projection head, we remove the projection head and train a classifier with a linear layer on top of it. This new model requires far fewer labeled examples to converge and that’s the gist of it. 

So, to sum it all up, we leveraged the huge dataset that is usually available to us and trained a base learner using unsupervised learning. This base learner will learn the representation by using a cosine similarity loss or a different version of it. Once the base learner converges, remove the projection head and replace it with a linear classifier and train your classifier with a little amount of labeled data. 


## Benefits
This approach saves time and effort in labeling a significant chunk of data and in the end, gives you a classifier that’s almost as good as if you trained the model using a huge labeled dataset. 
