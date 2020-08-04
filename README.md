#Table of Contents
[What is a neural network](#h)  

<a name="h"/>

## What is a neural network?

Let’s take an example of House Price Prediction, the datasets with the size of the houses in square meters and you know the price of the houses and now you want to fit a function to predict the price of the house. If you use linear regression, let’s put a straight line to these data and we get a straight line, this line being the function for predicting the price of the house of the house size 

![](https://i.imgur.com/0QwD3LZ.png)

In neutral network the size of the house will be x and goes into a node, this little circle and then its outputs the price of the house which we call y. The above network consists of a single neuron and, implements function that drawn in the linear regression. So the neuron takes the input feature size and then computed the linear function and takes a max of zero and the output the estimated price. Some most used activation function are ReLU, which takes max of (0, z)

![](https://i.imgur.com/Tp2sVer.jpg)

A larger neural network is formed by taking many neurons and stacking in together. Example of densely connected neural network, every input features are connected with every one of the neuron. 
