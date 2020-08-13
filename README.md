# Table of Contents

[What is a neural network](#h)  
[Supervised learning with Neural Network](#s)   
[Why is Deep Learning taking off?](#d)   
[Logistic regression on Deep learning](#l)   
[Gradient Descent](#g)

<a name="h"/>

## What is a neural network?

Let’s take an example of House Price Prediction, the datasets with the size of the houses in square meters and you know the price of the houses and now you want to fit a function to predict the price of the house. If you use linear regression, let’s put a straight line to these data and we get a straight line, this line being the function for predicting the price of the house of the house size 

![](https://i.imgur.com/0QwD3LZ.png)

In neutral network the size of the house will be x and goes into a node, this little circle and then its outputs the price of the house which we call y. The above network consists of a single neuron and, implements function that drawn in the linear regression. So the neuron takes the input feature size and then computed the linear function and takes a max of zero and the output the estimated price. Some most used activation function ReLU, which takes max of (0, z)

![](https://i.imgur.com/Tp2sVer.jpg)

A larger neural network is formed by taking many neurons and stacking in together. Example of densely connected neural network, every input features are connected with every one of the neuron. 

<a name="s"/>

## Supervised Learning

In supervised learning you have some input x and you want to learn a function mapping to some output y, In case house price prediction, we have some input feature size of house and want to predict the price of the house for size of house. 

Application of neural network.

![](https://imgur.com/J4gVzWO)

For sequential data we used Recurrent neural network (RNN) and for image application we used convolutional neutral network

![](https://i.imgur.com/F9u95J0.png)

Neural network works well with interpreting the unstructured data, create applications that use speech recognition, image recognition.

<a name="d"/>

## Why is Deep Learning taking off?

Scale drives of deep learning process is the large amount labelled samples, performance depends on Data, Computation, Algorithms.  

Algorithms
Breakthrough is switching from Sigmoid function to ReLU function. drawbacks of sigmoid function are the slope of the function would gradient is nearly zero, so that the learning is slow because when you implement gradient descent and gradient is zero the parameters just change very slowly. For Relu the gradient is equal to one for all positive values of input right and less likely gradient shrink to zero.

<a name="l"/>

## Logistic regression for Deep learning

Logistic regression is for binary classification, consider we have an image and we want to recognize this image as either being a cat, in case cat then we denote the output as 1 and not cat output as 0. 

![](https://imgur.com/dnGMuCT.png)

The computer store images as three separate matrices corresponding to red green and blue colors channels. So, if your image is 64 pixels by 64 pixels then you have 3 64 by 64 matrixes corresponding to the red green and blue. We convert these pixel intensity values into feature vector x, so single vector represents all the pixel intensity values, the total dimension of the vector x will be 64 by 64 multiplied by 3, in this case its turn out to be 12,228

![](https://imgur.com/JwLaeAO.png)


In logistic regression, output labels Y in a supervised learning problem are all zero or one, so for binary classification problems. Given input feature vector X corresponding to an image that you want to recognize an image cat or not, so you want an algorithm to predict the cat or not based on the input features, the output we will call as **Y hat**, which is the estimate of Y.

Given <img src="https://i.upmath.me/svg/x" alt="x" /> is a feature vector, <img src="https://i.upmath.me/svg/%5Chat%7By%7D%3D%20P(y%3D1%2Fx)%2C%20%5Chat%7By%7D" alt="\hat{y}= P(y=1/x), \hat{y}" /> is the estimate of y 

Parameters: <img src="https://i.upmath.me/svg/%20w%20%5C%20%5Cvarepsilon%20%20%5C%20%7B%5Crm%20I%5C!R%5En%5Ex%7D%20" alt=" w \ \varepsilon  \ {\rm I\!R^n^x} " />, <img src="https://i.upmath.me/svg/%20%20b%20%5C%20%5Cvarepsilon%20%5C%20%7B%5Crm%20I%5C!R%7D%20" alt="  b \ \varepsilon \ {\rm I\!R} " />

Output <img src="https://i.upmath.me/svg/%5Chat%7By%7D%3D%20%5Csigma%20(%7BW%5ETx%20%2B%20b%7D)%20%2F%20%5Csigma" alt="\hat{y}= \sigma ({W^Tx + b}) / \sigma" /> is a sigmoid function

### Logistic Regression cost function

loss error function: this will measure the how good the <img src="https://i.upmath.me/svg/%5Chat%7By%7D" alt="\hat{y}" />
 is when true label is y. square error is not a reasonable choice because it will make the gradient descent not work well. 


<img src="https://render.githubusercontent.com/render/math?math=L(\hat{y},y) = -(y\log \hat{y} %2B (1-y)\log(1-\hat{y})">

![](https://imgur.com/jGQFJSE.png)

<a name="g"/>

## Gradient Descent

The loss function measures how well the you’re doing the training, also measure how well the parameter w and b doing in the entire training set. 


<img src="https://i.upmath.me/svg/%5Chat%7By%7D%3D%20%5Csigma%20(%7BW%5ETx%20%2B%20b%7D)%2C%20%5C%20%20%5Csigma(z)%20%3D%20%7B1%20%5Cover%201%20%2B%20e%5E-%5Ez%7D%20" alt="\hat{y}= \sigma ({W^Tx + b}), \  \sigma(z) = {1 \over 1 + e^-^z} " />

the cost function for entire training set 

![](https://imgur.com/04pvAgW.png)

<img src="https://i.upmath.me/svg/J(W%2Cb)%3D%20-%7B1%20%5Cover%20m%20%7D%20%5Csum_%7Bn%3Di%7D%5E%7Bm%7D%20%7B%5By%5Ei%5Clog%5Chat%7By%7D%5Ei%7D%20%2B%20(1-y%5Ei)log(1-%5Chat%7By%7D%5Ei)%5D%20" alt="J(W,b)= -{1 \over m } \sum_{n=i}^{m} {[y^i\log\hat{y}^i} + (1-y^i)log(1-\hat{y}^i)] " />

We want to find w, b that minimize J(w,b) 
We step in, on each iteration find the parameters to minimise the cost function.

![](https://imgur.com/L3y4LCP.png) 

dw will be used to represent this derivative term. w get updated and represented as 
<img src="https://i.upmath.me/svg/w%3A%3D%20w%20-%5Calpha%20%5C%20%7B%20%5Cpartial%20T(w%2Cb)%20%5Cover%20%5Cpartial(w)%7D" alt="w:= w -\alpha \ { \partial T(w,b) \over \partial(w)}" /> 

The definition of a derivative is the slope of a function at the point. So the slope of the function is really the height divided by the width. 

<img src="https://i.upmath.me/svg/b%3A%3D%20w%20-%5Calpha%20%5C%20%7B%20%5Cpartial%20T(w%2Cb)%20%5Cover%20%5Cpartial(b)%7D" alt="b:= w -\alpha \ { \partial T(w,b) \over \partial(b)}" />


So in logistic regression we updating w as w minus the learning rate times the derivative of J(w,b) respect to w
you update b as b minus the learning rate times the derivative of the cost function in respect to b

### Derivatives
![](https://imgur.com/slq44w9.png)

slope is defined as the height divided by the width little triangle.

Derivatives are defined with an even smaller value of how much you nudge a to the right. So, it's not 0.001. It's not 0.000001. It's not 0.00000000 and so on 1. It's even smaller than that, and the formal definition of derivative says, whenever you nudge a to the right by an infinitesimal amount, basically an infinitely tiny, tiny amount

#### back propagation

![](https://imgur.com/41NelOW.png)
 Well, let's go through the example, where now a = 5. So let's bump it up to 5.001. The net impact of that is that v, which was a + u, so that was previously 11. This would get increased to 11.001. And then we've already seen as above that J now gets bumped up to 33.003. So what we're seeing is that if you increase a by 0.001, J increases by 0.003. And by increase a, I mean, you have to take this value of 5 and just plug in a new value. Then the change to a will propagate to the right of the computation graph so that J ends up being 33.003. And so the increase to J is 3 times the increase to a. So that means this derivative is equal to 3. And one way to break this down is to say that if you change a, then that will change v
 
 So dv/da = 1. So in fact, if you plug in what we have wrapped up previously, dv/dJ = 3 and dv/da = 1. So the product of these 3 times 1, that actually gives you the correct value that dJ/da = 3
 
![](https://imgur.com/DMlLBhk.png)

Derivation of gradient descent
![](https://imgur.com/jUhJw4l.png)