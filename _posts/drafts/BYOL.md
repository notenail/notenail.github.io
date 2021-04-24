As an engineer developing learning-based architecture in the field of robotics and automation, data generation is one of the easier challenges to overcome. But it is one thing to generate data in heaps and piles and a completely different thing to get a workable machine learning model out of it. 

Now, if you choose to develop your models using supervised learning, you’ll have to face the daunting task of completely annotating the data. Data annotation or data labeling is one of the primary consumer of capital, energy, and most importantly, time. Hence, supervised learning is the last thing I consider while designing a solution as the CTO of a startup in the times of COVID. 

So if not supervised, what should you do? 

The answer to this question includes semi-supervised, self-learning, and meta-learning.

In this blog post, I’ll be describing my implementation of the state-of-the-art method for self-supervised learning - BYOL: Bootstrap Your Own Latent.
