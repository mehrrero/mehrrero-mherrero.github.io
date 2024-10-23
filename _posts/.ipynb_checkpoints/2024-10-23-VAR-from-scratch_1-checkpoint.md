---
title: "Vector auto-regression from scratch: part 1"
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - Machine learning
  - Tutorials
---

Predicting the future values of a time series is a common task in the field of machine learning, with a lot of useful applications --- from the evolution of the stock market, electricity demand, traffic flow, ... to more *scientific* tasks, such as predicting the behavior of a physical system as time evolves. 

Vector auto-regression (VAR) is a common method to perform such predictions, by proposing a model that connects the value of a point in the time series to previous points in a linear manner. It has the advantage of being easy to write, with a number of parameters which scales roughly as $N^2$, where $N$ is the number of different time series that we aim to predict; and allows for interaction between all them. They might be sub-optimal when $N$ is large, or in the presence of strong stochastic behavior of the values to predict, but their core construction can be ported to more sophisticated models that aim to solve or relax these issues, such as matrix factorization models, or Bayesian improvements.

There are many packages in python where VAR algorithms are already implemented and can be used as a black box. However, for pedagogical reasons (and because I am also learning as I write this!), I will show here how to construct a VAR model from scratch and discuss some mathematical details about continuous functions that seem to justify the effectivenes and succes of methods based on vector auto-regression, despite their simplicity.

<!--more-->

I have been working for months in time series forecasting for gravitational waves, but in our research we never touched auto-regression. Instead, we were just simply predicting the full future outcome of a time series given its early part as a whole. This problem sounds similar to what auto-regression does, but it can be solved with more standard techniques. Instead of predicting every new value from its close neighbors, we were just solving a standard machine learning problem, where we have input data as vectors of a given dimension, and targets with a different dimension but also in vector form. Autoregression has always catched my interest, but I never had any time to invest in learning its intricacies properly. Until now. 

What you are reading is my **first** approach to autoregression, mainly fueled by a Kaggle competition that I just joined. As such, expect the code shown along this series of posts to be simple and straight to the point. The aim here is not to build an extremely sophisticated program, but instead to understand in detail what is happening in the inside of your computer when you tell it to perform vector autoregression. 

### The problem at hand

Let us start by defining the problem that we aim to solve. We have a set of $N$ time series, which depend on time and, perhaps, on a set of exogeous features $x_a(t)$, which is just a fancy name to label any other variable which might also change with time, but which we will consider here simply as an input for our target mathematical functions
$$y_i = y_i (x_a(t),t). $$

Here we are using tensor notation to denote the existence of several objects of the same kind. Hence, $y_i$ denotes the set of $N$ time series, with $i\in \{1,N\}$, and $x_a(t)$ represents the $F$ features that enter as aditional variables in the function $y_i$, with $a\in \{1,F\}$. 

The recipe for VAR estates that we can predict future values of $y_i(x_a(t),t)$ by using the ansatz
$$ y_i(x_a(t),t) = c_i + \sum_{k=1}^\delta  A^k_{ji} y_j(x_a(t-kh),t-kh) + \sum_{k=1}^\lambda  B^k_{ai} x_a(t-kh) + U_i(t),$$
where we are using the shorthand notation where repeated indices imply summation, i.e. $Y_a X_a = \sum_a Y_a X_a $. 

Here, $\delta$ and $\lambda$ denote the number of past steps $y$ and $x$ that we use to perform a prediction at the point $t$, while $h$ is the separation in between temporal points, that we have taken to be constant. For instance, for $\delta=2$ and $\lambda=1$, the previous ansatz becomes
$$y_i(x_a(t),t) = c_i + A^1_{ji} y_j(x_a(t-h),t-h) + A^2_{ji} y_j(x_a(t-2h),t-2h) + A^2_{ji} y_j(x_a(t-2h),t-2h) +  B^1{ai} x_a(t-h) + B^2{ai} x_a(t-2h)+ U_i(t). $$
Finally, $c_i$ is just a constant term and $U_t$ is a white noise. We will go back to it later, but we can ignore it for now.

### Some mathematical points

The previous anstaz for the evolution of our set of time series seems extremely simple. Indeed it consists only on linear combinations of the value of the function and its input parameters in previous time points. Note that there is no presence of any non-linearity, such as those that we find in the activation functions in neural networks. Hence, it is reasonable to ask ourselves how this simple formula is able to capture complicated non-linear behaviors. Well, I do not know the full mathematical foundation of VAR, but I can argue that this ansatz is general enough by invoking the mother of all modern science --- the theory of continuous functions and its main avatar, Taylor's theorem.

As you may know, any continuous function is expandable in an infinite series of terms of the form

$$ y_i(x(t),t) = y_i(x(t_0),t_0) + \sum_{k=1}^\infty \sum_{j=1}^\infty \frac{\partial^k y_i}{\partial t^k} \frac{\partial^k y_i}{\partial x_a(t)^j}(t-t_0)^k (x_a(t)-x_a(t_0))^j. $$
where the derivatives are evaluated at $t=t_0$.

This contains a constant term $y_i(x(t_0),t_0)$, corresponding to the value of the time series at the collocation point $t_0$, and terms with derivatives. This is the key point here. I am not going to enter into details of why this is possible, but any derivative of a continuous function can always be written in terms of *linear combinations* [of the value of the function](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119083405.app1) along previous (or future, but this is useless for us) points in time. For instance, a second derivative can be written as
$$\frac{d^2 y_i(x(t),t)}{dt^2} \sim \frac{2 y(x(t_0),t_0)-5y(x(t_0-h),t_0-h)+4y(x(t_0-2),t_0-2)-y(x(t_0-3),t_0-3)}{h^2} $$.

Thus, we can always think of the Taylor expansion in the previous formula as a linear combination of the values of $y_i(x(t),t)$ and $x(t)$ evaluated along a temporal grid, which *is precisely what the VAR ansatz does*. Hence, Taylor's theorem for a continuous function seems to give a good justification on why this ansatz works. In there, the matrices $A^k_{ij}$ and $B^k_{ij}$ seem to be parametrizing the unknown coefficients which would come with the expansion of both $y(x(t),t)$ and $x(t)$, along with, perhaps, other effects that we might ignore in our simple discussion here.

### Stochasticity

After this formal detour, let us discuss the last element in the VAR anstaz, the vector $U_i(t)$. This element here is key in improving VAR performance for real world application, where data has, most of the time, a stochastic behavior due to underlying random features. Stock markets depend on human behavior, weather measures vary with thermal fluctuations, and health data might be affected by a million of external factors. Hence, we should build our models to be non-deterministic, and instead contain an inner stochasticity that allows us to compute a mean and standard deviation for our predictions. This is provided in the case at hand by the vector $U_i(t)$ which, at each time point is sampled from a multivariate Gaussian distribution $M{\cal N}(0, \tau_i^{-1})$, where we assume that its different elements are independent.

### It's time to code

Ok, this explanation is great, I know (hats off), but we are now interested in putting all this in play and really build a model which uses VAR to fit some time series data and predict subsequent points. How do we do it?

The first step is to build our core model, implementing the ansatz shown before. I do it here using Pytorch

```
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class VAR(nn.Module):
    '''
    Defines a VAR layer.

    Parameters:
        N (int): Number of time series.
        F (int): Number of features.
        lag_y (int): Number of points in the past to consider for prediction.
        lag_f(int): Number of features in the past to consider for prediction.

    '''

    def __init__(self, N, F, lag_y, lag_f):
        super(VAR, self).__init__()

        self.C = nn.Parameter(torch.rand((N,1)))
        self.C.requires_grad = True
        self.A = nn.ParameterList([])                      
        self.B = nn.ParameterList([])
        self.A.requires_grad = True
        self.B.requires_grad = True
        for i in range(lag_f+1):
            self.B.append(nn.Parameter(torch.rand((N,F))))
        
        for i in range(lag_y):
            self.A.append(nn.Parameter(torch.rand(N,N)))

        for x in self.A:
            x.requires_grad = True

        for y in self.B:
            y.requires_grad = True
       
        self.tau = nn.Parameter(torch.rand(N))
        self.tau.requires_grad = True
        
        self.lag_y = lag_y
        self.lag_f = lag_f
        self.N = N
        self.F = F
        
    def forward(self, y_t_lag, f_t_lag):
        '''
        Computes the value of the N outputs of the model by using lag_y previous points and lag_f previous features.

        Parameters:
            y_t_lag (tensor, shape = (lag_y, N ,1)): Values of the N time series in the previous lag_y points.
            f_t_lag (tensor, shape = (lag_f, F, 1)): Values of the F features in the previous lag_f points.

        Output:
            y_t1 (tensor, shape = (N,1)): Prediction at the next time point.

        '''
  
        y_t1 = torch.zeros(self.N,1)
        y_t1 += self.C + torch.normal(mean=0.0, std=1/self.tau).reshape(self.N,1)
        
        for i in range(self.lag_y):
            y_t1 += torch.matmul(self.A[i],y_t_lag[i])
            
        for i in range(self.lag_f+1):
            y_t1 += torch.matmul(self.B[i],f_t_lag[i])

        return y_t1 


```

We have done a few things here. First, we have imported the libraries needed for this class and for the next code that we will discuss. Second, we have defined our VAR model as a child class from the nn.Module in Pytorch, which allows us to inherit all the methods and properties of the parent class. Finally, we have defined a custom forward method which predicts a point given a set of previous values of $y$ and $x$ (called here $f$, from features). The trainable parameters of the model are the matrices $A$ and $B$, as well as the vector $C$ and the set of precision values $\tau$ that enter into $U$. Note that we have explicitly demanded `requires_grad = True` in all parameters, so Pytorch can build a gradient graph for them when training the model.

The next step is to wrap this into another class, where we can implement the training loop and the rest of functions that we need to make it work. This step is not needed in principle. You could do everything with functions, or even adapting the standard Pytorch workflow, but I like it better in this form since it produces a much cleaner and readable code. However, this will have to wait for the next entry of this post series.











































