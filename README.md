# Table of Contents

[What is a neural network](#h)  
[Supervised learning with Neural Network](#s)   
[Why is Deep Learning taking off?](#d)   
[Logistic regression on Deep learning](#l)

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









