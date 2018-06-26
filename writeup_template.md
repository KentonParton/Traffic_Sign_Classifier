# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic Summary of Dataset

I used the built in Python .len function and Pandas .shape function to generate a statistics summary for the traffic
sign data set:

#### The following stats were calculated:
* The size of training set is: 34,799
* The size of the validation set is: 4,410
* The size of test set is: 34,799
* The shape of a traffic sign image is: (32, 32, 1)
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Initially, the images were going to be converted to grayscale; however, because color can be considered a feature of road signs they were left in color. While this would require a longer training time, it would result in a more accurate classification model.

While the images were kept in color, they were normalize using Min-Max Scaling to a range of [0.1, 0.9]. By normalizing the image data it reduced training time and increased the accuracy of the model........


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         	|     Description	        					    | 
|:-----------------:|:-------------------------------------------------:| 
| Input             | 32x32x3 RGB image                                 | 
| Convolution 1x1   | Stride: 1x1, Padding: valid, Outputs: 32x32x32    |
| RELU				|												    |
| Convolution 3x3   | Stride: 1x1, Padding: valid, Outputs: 30x30x32    |
| RELU				|												    |
| Convolution 3x3   | Stride: 1x1, Padding: valid, Outputs: 28x28x64    |
| RELU				|												    |
| Max pooling	    | Stride: 2x2, Kernel: 1x1, outputs: 14x14x64 	    |
| Dropout			| Drop Rate: 50%								    |
| Convolution 3x3   | Stride: 1x1, Padding: valid, Outputs: 12x12x128   |
| RELU				|												    |
| Convolution 3x3   | Stride: 1x1, Padding: valid, Outputs: 10x10x256   |
| RELU				|												    |
| Max pooling	    | Stride: 2x2, Kernel: 1x1, outputs: 5x5x64 	    |
| Dropout			| Drop Rate: 50%								    |
| Flatten			|               							        |
| Fully connected	| Input: 256, Output: 84         					|
| Fully connected	| Input: 84, Output: 43         					|
| Softmax			|           									    |


My model followed the notion that CNN's need to be deeper rather than wider. This was done by using a series of 3x3 convolutional layers which increased in depth with each proceeding convolution. This approach was based on the paper(...) which demonstrated that(...) This allowed the network to detect more detailed features in the images.

To reduce the number of parameters, a 1x1 convolution was used in the first layer, acting as another form of preproccessing to reduce the training time of the model. According to (...a 1x1 convolution in the first layer has ...) Furthermore, two dropout layers at 50% were used to prevent the model from overfitting and to promote new features being detected. 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, a library named (...) was used to perform a series of training iterations to test for the best hyperperameters such as learning-rate and batch size; it was also used to test different network structures. Based on the training iterations, the network performed best with two dropout layers directly after the two max pooling layers. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

First, a base case was established to create a MVP, which in this case was the LeNet architecture. After the LeNet architecture was edited to suit the traffic sign data, a series of tests and architectural changes were made to LeNet, specifically the output depth and kernel size for each convolutional layer. After minor changes to the architecture and adding a dropout layer, a validation accuracy of 95% was acheived.

After this, implementations of (...) architectures were implemented to identify the best one. 

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The LeNet architecture was was adjusted and used to test and gain a better understanding of the project.

* What were some problems with the initial architecture?
With the default data, a maximum validation accuracy of 95% could be achieved and was not high enough; therefore, new architectures were tested.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Initially, the kernel size and convolutional output depth was adjusted to test of a higher validation accuracy could be acheived. After reaching a maximum accuracy of 95%, new architectural models were tested.

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


