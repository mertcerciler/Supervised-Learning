import numpy as np
import matplotlib.pyplot as plt

#below functions are functions that is used both ann algorithms or functions that is used by main

#sigmoid function.
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#derivative of sigmoid function.
def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

#function for reading and seperating datasets into inputs and labels.                
def reading_datasets(f):
    x = []
    y = []
    for line in f:
        is_x  = True
        x_index = ''
        y_index = ''
        for ch in line:
            if (ch == '\t'):
                is_x = False
            elif(ch == '\n'):
                x_index = float(x_index)
                x.append(x_index)
                y_index = float(y_index)
                y.append(y_index)
                break
            if is_x:
                x_index += ch
            else:
                y_index += ch
    x = np.array(x)
    y = np.array(y)    
    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0], 1)
    return x, y

#returns loss of the output, with the sum of squares loss function.
def loss(output_set, label_set):
    loss = np.square(output_set - label_set)
    return sum(loss)/output_set.shape[0]

def standard_dev(output_set, label_set):
    loss = np.square(output_set - label_set)
    return np.std(loss)

#returns the normalized dataset.
def normalization(x_train, y_train, x_test, y_test):
    #first, concatenate all datasets to a single dataset in order to make normalization according to whole dataset. 
    all_dataset = np.concatenate((x_train, y_train, x_test, y_test), axis=0)
    norm_x_train = x_train
    norm_y_train = y_train
    norm_x_test = x_test
    norm_y_test = y_test 
    #normalize x_train 
    for row in range(0, x_train.shape[0]):
        norm_x_train[row] = (x_train[row] - all_dataset.min()) / (all_dataset.max() - all_dataset.min()) 
    #normalize y_train    
    for row in range(0, y_train.shape[0]):
        norm_y_train[row] = (y_train[row] - all_dataset.min()) / (all_dataset.max() - all_dataset.min())
    #normalize x_test    
    for row in range(0, x_test.shape[0]):
        norm_x_test[row] = (x_test[row] - all_dataset.min()) / (all_dataset.max() - all_dataset.min()) 
    #normalize y_test    
    for row in range(0, y_test.shape[0]):
        norm_y_test[row] = (y_test[row] - all_dataset.min()) / (all_dataset.max() - all_dataset.min()) 
    
    return norm_x_train, norm_y_train, norm_x_test, norm_y_test

def plot(x_points, real_values, predictions, title, xlabel, ylabel):
    fig = plt.figure() 
    ax1 = fig.add_subplot(111)
    ax1.scatter(x_points, real_values, s=10, c='b', marker="s", label="Real Values")
    ax1.scatter(x_points, predictions, s=10, c='r', marker="s", label="Predictions")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
#This is the class for ann with single hidden layer.
class ann_with_shl: 
    def __init__(self, x, y, units, max_epoch, learning_rate, stop_value):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],units) 
        self.weights2   = np.random.rand(units,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.max_epoch  = max_epoch
        self.step_size1 = 1
        self.step_size2 = 1
        self.learning_rate = learning_rate
        self.training_output = np.zeros(self.output.shape[0])
        self.stop_value = stop_value
     
    #feedforward algorithm.
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))    
        self.output = np.dot(self.layer1, self.weights2)
    
    #backpropagation algorithm.
    def backpropagation(self):
        loss_derivative = 2*((self.output-self.y))
        sgm_derivative = sigmoid_derivative(self.layer1)
        
        #chain rule.
        gradient_hidden2output = np.dot(self.layer1.T, loss_derivative)
        gradient_input2hidden = np.dot(self.input.T, np.dot(loss_derivative, self.weights2.T) * sgm_derivative)
        
        #calculating step size with multiplying gradient with learning rate.
        self.step_size2 = gradient_hidden2output * -1*self.learning_rate
        self.step_size1 = gradient_input2hidden * -1*self.learning_rate
        #print('stepsize1:', self.step_size1.shape)
        
        #updating the weights. 
        self.weights2 = self.step_size2 + self.weights2
        self.weigths1 = self.step_size1 + self.weights1
        
    #fitting the dataset to ann with single layer with given conditions.
    def fit(self):
        epoch = 1
        error_training = np.zeros(self.max_epoch)
        while (epoch <= self.max_epoch or (self.step_size1.any() <= self.stop_value and self.step_size2.any() <= self.stop_value)):
            self.feedforward()
            if epoch == self.max_epoch:
                self.training_output = self.output
            self.backpropagation()
            loss_fitting = loss(self.output, self.y)
            error_training[epoch-1] = loss_fitting
            epoch += 1
            stdev = standard_dev(self.output, self.y)
        return stdev, self.training_output, error_training
    
    #predicting the given dataset and calculating the error
    def predict(self, input_set, label_set):
        hidden_layer = sigmoid(np.dot(input_set, self.weights1))
        prediction  = np.dot(hidden_layer, self.weights2)
        loss_prediction = loss(prediction, label_set)
        error_test = loss_prediction
        return prediction, error_test
    
                
#This is the class for ann with linear regressor.      
class ann_with_lr:
    def __init__(self, x, y, max_epoch, learning_rate, stop_value):
        self.input = x
        self.weights = np.random.rand(self.input.shape[1],1)
        self.y = y
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.step_size = 1
        self.stop_value = stop_value
        self.output = np.zeros(self.y.shape)
        self.training_output = np.zeros(self.output.shape[0])
    
    #feedforward algorithm.
    def feedforward(self):
        self.output = np.dot(self.input, self.weights)
    
    #backpropagation algorithm.
    def backpropagation(self):
        loss_derivative = 2*((self.output - self.y))

        #chain rule.
        gradient = np.dot(self.input.T, loss_derivative)
        
        #calculating step size with multiplying gradient with learning rate.
        self.step_size = gradient * -1*self.learning_rate
        
        #updating the weights.
        self.weights = self.step_size + self.weights
    
    #fitting the dataset to ann with single layer with given conditions.    
    def fit(self):
        epoch = 1 
        error_training = np.zeros(self.max_epoch)
        while (epoch <= self.max_epoch or (self.step_size.any()  <= self.stop_value)):
            self.feedforward()
            if epoch == self.max_epoch:
                self.training_output = self.output
            self.backpropagation()
            loss_fitting = loss(self.output, self.y)
            error_training[epoch-1] = loss_fitting
            epoch +=1
            stdev = standard_dev(self.output, self.y)
        return stdev, self.training_output, error_training
    
    #predicting the given dataset and calculating the error
    def predict(self, input_set, label_set):
        prediction = np.dot(input_set, self.weights)
        loss_prediction = loss(prediction, label_set)
        error_test = loss_prediction
        return prediction, error_test
       
#main (we are initilazing and calling the classes below).
        
#reading datasets and seperate them into inputs and outputs.    
f_training = open("train1.txt", "r")
f_test = open("test1.txt", "r")
x_train, y_train = reading_datasets(f_training)
x_test, y_test = reading_datasets(f_test)

#Normalize the whole dataset. Normalization do not be applied since learning process can be observed if the numbers are bigger. 
#If it is wanted to normalize the dataset, below line should be commented out.
#x_train, y_train, x_test, y_test = normalization(x_train, y_train, x_test, y_test)

#training the dataset with ann single hidden layer with 2 hidden units.
ann_shl_2 = ann_with_shl(x = x_train, y = y_train, max_epoch = 100, units = 2, learning_rate = 0.025, stop_value = 0.0001)
stdev_2, training_output_shl_2, error_training_shl_2 = ann_shl_2.fit()
prediction_shl_2, error_test_shl_2 = ann_shl_2.predict(x_test, y_test)

#training the dataset with ann single hidden layer with 4 hidden units.
ann_shl_4 = ann_with_shl(x = x_train, y = y_train, max_epoch = 100, units = 4, learning_rate = 0.01, stop_value = 0.0001)
stdev_4, training_output_shl_4, error_training_shl_4 = ann_shl_4.fit()
prediction_shl_4, error_test_shl_4 = ann_shl_4.predict(x_test, y_test)

#training the dataset with ann single hidden layer with 8 hidden units.
ann_shl_8 = ann_with_shl(x = x_train, y = y_train, max_epoch = 100, units = 8, learning_rate = 0.001, stop_value = 0.0001)
stdev_8, training_output_shl_8, error_training_shl_8 = ann_shl_8.fit()
prediction_shl_8, error_test_shl_8 = ann_shl_8.predict(x_test, y_test)

#training the dataset with ann single hidden layer with 16 hidden units.
ann_shl_16 = ann_with_shl(x = x_train, y = y_train, max_epoch = 100, units = 16, learning_rate = 0.0015, stop_value = 0.0001)
stdev_16, training_output_shl_16, error_training_shl_16 = ann_shl_16.fit()
prediction_shl_16, error_test_shl_16 = ann_shl_16.predict(x_test, y_test)

#training the dataset with ann single hidden layer with 32 hidden units.
ann_shl_32 = ann_with_shl(x = x_train, y = y_train, max_epoch = 100, units = 32, learning_rate = 0.001, stop_value = 0.0001)
stdev_32, training_output_shl_32, error_training_shl_32 = ann_shl_32.fit()
prediction_shl_32, error_test_shl_32 = ann_shl_32.predict(x_test, y_test)

#training the dataset with ann linear regressor
ann_lr = ann_with_lr(x = x_train, y = y_train, max_epoch = 100, learning_rate = 0.0005, stop_value = 0.00001)
stdev_lr, training_output_lr, error_training_lr  =ann_lr.fit()
prediction_lr, error_test_lr = ann_lr.predict(x_test, y_test)

#plotting the predictions vs labels.
x_points_test = np.zeros(y_test.shape[0])
for i in range(0, y_test.shape[0]):
    x_points_test[i] = i
    
x_points_train = np.zeros(y_train.shape[0])
for i in range(0, y_test.shape[0]):
    x_points_train[i] = i
   
plot(x_points=x_points_test, real_values=y_test, predictions=prediction_lr, 
     title='Test predictions vs test labels for linear regressor', xlabel='Input points', ylabel='Output points')
plot(x_points=x_points_train, real_values=y_train, predictions=training_output_lr, 
     title='Training outputs vs traning labels for linear regressor', xlabel='Input points', ylabel='Output points')
plot(x_points=x_points_train, real_values=y_train, predictions=training_output_shl_2, 
     title='Training outputs vs traning labels for ann with shl with 2 hidden units', xlabel='Input points', ylabel='Output points')
plot(x_points=x_points_train, real_values=y_train, predictions=training_output_shl_4, 
     title='Training outputs vs traning labels for ann with shl with 4 hidden units', xlabel='Input points', ylabel='Output points')
plot(x_points=x_points_train, real_values=y_train, predictions=training_output_shl_8, 
     title='Training outputs vs traning labels for ann with shl with 8 hidden units', xlabel='Input points', ylabel='Output points')
plot(x_points=x_points_train, real_values=y_train, predictions=training_output_shl_16, 
     title='Training outputs vs training labels for ann with shl with 16 hidden units', xlabel='Input points', ylabel='Output points')
plot(x_points=x_points_train, real_values=y_train, predictions=training_output_shl_32, 
     title='Training outputs vs traning labels for ann with shl with 32 hidden units', xlabel='Input points', ylabel='Output points')










