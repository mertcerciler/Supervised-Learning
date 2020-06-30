# Supervised-Learning
 
ANN with linear regressor and ANN with single hidden layer is implemented and the dataset is applied to both models. 

* Sum of Squared errors are used as loss function.
* Sigmoid activation function is used to define the hidden units. 
* Stochastic learning algorithm is used. 

Different parameters are used to find the best model that gives the lowest error result. 

The resulting configuration is: 
* ANN used: ANN for linear regression gives more accurate results.
* Learning rate: 0.0005.
* I initialized weights with random numbers between (0,1) for both models. 
* I used different numbers of epochs with different parameters. The number is 100 in this configuration.
* The stop value is 0.00001. Even the epoch does not reach its max, if the step size is lower than the stop value, learning is terminated. 
* Normalization highly affects the error value since the difference of maximum and minimum values of the dataset is very high. If the normalization applied with respect to whole dataset, the numbers are decreased hugely. That makes the error value and the data points lower. However, I don’t think the ratio between the data points and the error values are also decreased, it is used for just playing with lower numbers. Since, learning process can be observed with bigger numbers, I choose not to apply the normalization. 
* **Training loss (averaged value over training instances): 1586**
* **Test loss (averaged value over test instances): 1275**
 

