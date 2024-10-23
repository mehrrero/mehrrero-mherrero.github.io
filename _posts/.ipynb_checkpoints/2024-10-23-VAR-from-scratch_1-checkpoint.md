---
title: "Vector auto-regression from scratch"
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - Machine learning
  - Tutorials
---

Predicting the future values of a time series is a common task in the field of machine learning, with a lot of useful applications --- from the evolution of the stock market, electricity demand, traffic flow, ... to more *scientific* tasks, such as predicting the behavior of a physical system as time evolves. 

Vector auto-regression (VAR) is a common method to perform such predictions, by proposing a model that connects the value of a point the time series to previous points in a linear manner. It has the advantage of being easy to write, with a number of parameters which scales roughly as $N^2$, where $N$ is the number of different time series that we aim to predict; and allows for interaction between all them. They might be sub-optimal when $N$ is large, or in the presence of strong stochastic behavior of the values to predict, but their core construction can be ported to more sophisticated models that aim to solve or relax these issues, such as matrix factorization models, or Bayesian improvements.

There are many packages in python where VAR algorithms are already implemented and can be used as a black box. However, for pedagogical reasons (and because I am also learning as I write this!), I will show here how to construct a VAR model from scratch and discuss some mathematical details about continuous functions that seem to justify the effectivenes and succes of methods based on vector auto-regression, despite their simplicity.

<!--more-->

I have been working for months in time series forecasting for gravitational waves, but in our research we never touched auto-regression. Instead, we were just simply predicting the full future outcome of a time series given its early part as a whole. This problem sound similar but it can be solved with more standard techniques. Instead of predicting every new value from its close neighbors, we were just solving a standard machine learning problem, where we have input data as vectors of a given dimension, and targets with a different dimension but also in vector form. Autoregression has always catched my interest, but I never had any time to invest in learning its intricacies properly. Until now. 

What you are reading is my **first** approach to autoregression, mainly fueled by a Kaggle competition that I just joined. As such, expect the code below to be simple and straight to the point. The aim here is not to build a extremely sophisticated program, but instead to understand in detail what is happening in the inside of your computer when you tell it to perform vector autoregression. 

### The problem at hand

Let us start by defining the problem that we aim to solve. We have a set of $N$ time series, which depend on time and, perhaps, a set of exogeous features, which is just a fancy name to label any other variable which might also change with time but which we will consider as an input for our target mathematical functions
$$y_i = y_i (x_a(t),t). $$

Here, we are using tensor notation to denote the existence of several objects of the same kind. Hence, $y_i$ denotes the set of $N$ time series, with $i\in \{1,N\}$, and $x_a(t)$ represents the $F$ features that enter as aditional variables in the function $y_i$, with $a\in \{1,F\}$. 

The recipe for VAR estates that we can predict future values of $y_i(x_a(t),t)$ by using the ansatz
$$ y_i(x_a(t),t) = c_i + \sum_{k=1}^\delta  A^k_{ji} y_j(x_a(t-kh),t-kh) + \sum_{k=1}^\lambda  B^k_{ai} x_a(t-kh) + U_i(t),$$
where we are using the shorthand notation where repeated indices imply summation, i.e. $Y_a X_a = \sum_a Y_a X_a $. 

Here, $\delta$ and $\lambda$ denote the number of steps back for $y$ and $x$ that we used to perform a prediction at the point $t$, while $h$ is the separation in between temporal points, that we have taken to be constant. For instance, for $\delta=2$ and $\lambda=1$, the previous ansatz becomes
$$y_i(x_a(t),t) = c_i + A^1_{ji} y_j(x_a(t-h),t-h) + A^2_{ji} y_j(x_a(t-2h),t-2h) + A^2_{ji} y_j(x_a(t-2h),t-2h) +  B^1{ai} x_a(t-h) + B^2{ai} x_a(t-2h)+ U_i(t). $$
Finally, $c_i$ is just a constant term and $U_t$ is a white noise. We will go back to it later, but we can ignore it for now.

### Some mathematical points

The previous anstaz for the evolution of our set of time series seems extremely simple. Indeed it consists only on linear combinations of the value of the function and its input parameters in previous time point, with no presence of any non-linearity, such as those that we find in the activation functions in neural networks. Hence, it is reasonable to ask ourselves how this is able to capture complicated non-linear behaviors. Well, I do not know the full mathematical foundation of VAR, but I can argue that this ansatz is general enough by invoking the mother of all modern physics --- the theory of continuous functions and its main avatar, Taylor's theorem.

As you may know, any continuous function is expandable in an infinite series of terms of the form
$$ y_i(x(t),t) = y_i(x(t_0),t_0) + \sum_{k=1}^\infty \sum_{j=1}^\infty \left.\frac{\partial^k y_i}{\partial t^k}\right_{t=t_0}\left. \frac{\partial^k y_i}{\partial x_a(t)^j}\right_{t = t_0}(t-t_0)^k (x_a(t)-x_a(t_0))^j. $$

This contains a constant term $y_i(x(t_0),t_0)$, corresponding to the value of the time series at the collocation point t_0, and terms containing derivatives. This is the key point here. I am not going to enter into details of why this is possible, but any derivative of a continuous function can always be written in terms of *linear combinations* [of the value of the function](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119083405.app1) along previous (or future, but this is useless for us) points of the function. For instance, a second derivative can be written as
$$\frac{d^2 y_i(x(t),t)}{dt^2} \sim \frac{2 y(x(t_0),t_0)-5y(x(t_0-h),t_0-h)+4y(x(t_0-2),t_0-2)-y(x(t_0-3),t_0-3)}{h^2} $$.

Thus, we can always think of the Taylor expansion in the previous formula as a linear combination of the values of $y_i(x(t),t)$ and $x(t)$ evaluated along a temporal grid, which *is precisely what the VAR ansatz does*. Hence, Taylor's theorem for a continuous function seems to give a good justification on why this ansatz works. In there, the matrix coefficients $A^k_{ij}$ and $B^k_{ij}$ seem to be parametrizing the unknown coefficients which would come with the expansion of both $y(x(t),t)$ and $x(t)$, along with, perhaps, other effect that we might ignore in our simple discussion here.

### Stochasticity

After this formal detour, let us discuss the last element in the VAR anstaz, the vector $U_i(t)$. This element here is key in improving VAR performance for real workd application, where data has, most of the time, a stochastic behavior due to underlying random features. Stock markets depend on human behavior, weather measure vary with thermal fluctuations, and health data might be affecte by a million of external factors. Hence, we expect our models to be non-deterministic and instead contain an inner stochasticity that allows us to compute a mean and standard deviation for our predictions. This is provided by the vector $U_i(t)$ which, at each time point is sampled from a multivariate Gaussian distribution $M{\cal N}(0, \tau_i^{-1})$, where we assume that its different elements are independent.


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


We have done a few things here. First, we have imported the libraries needed for this class and for the next one. Second, we have defined our VAR model as a class heritaged from the nn.Module in Pytorch, which allows us to inherit all the methods and properties of the parent class. Finally, we have defined a custom forward method which predicts a point given a set of previous values of $y$ and $x$ (called here $f$, from features). The trainable parameters of the model are the matrices $A$ and $B$, as well as the vector $C$ and the set of precision values $\tau$ that enter into $U$. Note that we have explicitly demanded `requires_grad = True` in all parameters, so Pytorch can build a gradient graph for them when training the model.

The next step is to wrap this into another class, where we can implement the training loop and the rest of function that we need to make it work. This step is not needed in principle. You could do everythign with functions, or even adapting the standard Pytorch workflow, but I like it better in this form since it produces a much cleaner and readable code. 

```
class trainer():
    '''
    Wrapper for the VAR class. Includes methods for evaluation and training.

    Parameters:
        N (int): Number of time series.
        F (int): Number of features.
        lag_y (int): Number of points in the past to consider for prediction.
        lag_f(int): Number of features in the past to consider for prediction.
        
    '''


    
    def __init__(self, N, F, lag_y, lag_f):
        super(trainer, self).__init__()
        self.var = VAR(N, F, lag_y, lag_f)
        self.lag_y = lag_y
        self.lag_f = lag_f
        self.N = N
        self.F = F
        
    def evaluate(self, y, f):
        '''
        Evaluates predictions for all times after lag_y, using the input data y, f for previous points.

        Parameters:
            y (tensor, shape = (T, N, 1)): Values of the input time-series at temporal points.
            f (tensor, shape = (T, F, 1)): Values of the feature vectors at temporal points.

        Output:
            target (tensor, shape = (T-y_lag, N ,1)): Predictions for the time points, excluding the first y_lag points.
        '''
        
        target = []
        for i in range(np.amax([self.lag_y,self.lag_f]),len(y)):
            yt = y.narrow(dim=0, start = i-self.lag_y, length=self.lag_y)
            ft = f.narrow(dim=0, start = i-self.lag_f, length = self.lag_f+1)
            target.append(self.var(yt,ft))
        return torch.stack(target)

    def loss(self,yp1, yp2,y):
        '''
        Evaluates the loss function, defined as -log(L), where L is the likelihood of the model.

        Parameters:
            yp1 (tensor, shape = (T-y_lag, N, 1)): A realization of the prediction.
            yp2 (tensor, shape = (T-y_lag, N, 1)): A second realization of the prediction.
            y (tensor, shape = (T, N, 1)): True values of the target time-series.

        Output:
            like (tensor, scalar): The value of -log(L).
        '''

        
        delta = len(y)-len(yp1)
        yr = torch.narrow(y, 0, delta, y.shape[0]-delta)
        y1 = (yr - yp1).flatten()
        y2 = (yr - yp2).flatten()
        # Stack the residuals
        residuals = torch.vstack((y1, y2))  # Shape: (2, len(y1))

        # Compute the covariance matrix of the residuals
        c = torch.cov(residuals)

        # Add a small value to the diagonal of covariance for stability (helps prevent singular matrix issues)
        c += torch.eye(2) * 1e-6  

        # Compute the inverse of the covariance matrix
        sigma_inv = torch.inverse(c)

        # Compute the distance (vectorized form)
        distances = torch.matmul(residuals.T, sigma_inv)  # Shape: (len(y1), 2)
        distances = torch.sum(distances * residuals.T, dim=1)  # Shape: (len(y1),)

        # Log-likelihood computation
        l2 = torch.sum(distances)  # Sum over the distances
        like = 0.5 * len(y1) * torch.logdet(c) + 0.5 * l2  # -Log determinant of the covariance matrix

        return like

    def train(self,epochs, y, f, learning_rate = 1e-3):
        '''
        Trains the model using stochastic gradient descent.

        Parameters:
            epochs (int): Number of training epochs.
            y (tensor, shape = (T, N, 1)): True values of the target time-series.
            f (tensor, shape = (T, F, 1)): Values of the vector of features.
            learning_rate (float, optional, default value = 1e-3): Learning rate.
        
        '''
        
        optimizer = optim.Adam(self.var.parameters(), lr=learning_rate)
        for epoch in trange(epochs):
            optimizer.zero_grad()
            yp1 = self.evaluate(y,f)
            yp2 = self.evaluate(y,f)
            loss = self.loss(yp1,yp2,y)
            loss.backward(retain_graph = True)
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

    def predict(self, y_i, f_i, f_p):
        '''
        Returns a prediction over a set of new temporal points.

        Parameters:
            y_i (tensor, shape = (Z, N, 1)): Values of the time-series previous to the prediction.
            f_i (tensor, shape = (Z, F, 1)): Values of the feature vector previous to the prediction.
            f_p (tensor, shape = (M, F, 1)): Values of the feature vector in the points where we predict.

        Output:
            y_f (tensor, shape = (M, N, 1)): Predictions on the M new points.
        
        '''
        pf = f_i.clone().detach()
        for f in f_p:
            _y = y_i[-self.lag_y:]
            _f = torch.cat((pf[-self.lag_f:],f.reshape(1,1,1)))
            pf = torch.cat((pf,f.unsqueeze(0)))
            y_i = torch.cat((y_i,self.var(_y,_f).unsqueeze(0)))
        return y_i[-len(f_p):]

```

This class is a bit more involved than the model that we defined before, as it contains a few different methods. First, we have added a method `evaluate` which produces predictions on the points of a given set, using that set to extract the previous data points. It is mean to be used in the training procedure, where we ask the model to produce data points by using the **true** previous values. Afterwards, the loss function evaluates the difference between this prediction and the real values. We do it here in a slightly spetial way that I will describe later in detail.

Finally, we can find a pretty standard (and simple!) training loop, and a method `predict`, which we use to produce subsequent values of the time series, using the prediction themselves as previous points to keep evolving the functions in time.

### Maximum likelihood method

In order to train the model, we need to target the minimum of a given loss function. There are many options that can be used. Standard approaches are based on minimizing the distance between the two datasets, the true values of our target variables, and the predictions, by using some distance metric in a given vector space. Common choices are the $L_1$ and $L_2$ norms, as well as some combinations of modifications of them. Here, however, we are dealing with a model which produces a stochastic result. In other words, every time we run the model, even if the set of input parameters has not changed, we will obtain a different result, whose statistics will be dictated by some underlying distribution which, in principle, we do not know.












































