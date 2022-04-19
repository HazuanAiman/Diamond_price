# Diamond Price Prediction using Feedforward Neural Network (FNN)
## Problem Statement
The aim of this project is to make a model that can predict the price of diamond based on the feature and data available. The dataset used can be obtained [here](https://raw.githubusercontent.com/HazuanAiman/Diamond_price/main/dataset/diamonds.csv)
<br>
<br>
Credit: [source](https://www.kaggle.com/datasets/shivam2503/diamonds)
## Methodology
#### IDE and Library
This project is made using Spyder as the main IDE. The main library used in this project are Pandas, Numpy, Scikit-learn and Tensorflow Keras.
#### Model Pipeline
In this project, a feedforward neural network is used and the output result is a regression problem. The activation function used is relu. Figure below shows the structure of the model created.
<p align="center">
  <img src="https://github.com/HazuanAiman/Diamond_price/blob/main/images/model%20pipeline.PNG">
<p>

The model is trained with a batch size of 32 and 100 epochs. Early stopping is applied in the training and it triggers at epochs 45/100. This is to prevent the model from overfitting. The training MAE achieved is 330 and the validation MAE is 323. Figures below show the graph of the training process.
<p align="center">
  <img src="https://github.com/HazuanAiman/Diamond_price/blob/main/images/epoch%20mae.PNG">
<p>
<p align="center">
  <img src="https://github.com/HazuanAiman/Diamond_price/blob/main/images/epoch%20loss.PNG">
<p>
  
## Results
The model is trained using the train dataset and evaluated using the test dataset in the train-test split. The test result are as show below:
<p align="center">
  <img src="https://github.com/HazuanAiman/Diamond_price/blob/main/images/result.PNG">
<p>

The graph of predictions vs labels of the test data was also plotted. The trendline of y=x can be seen which means that the predictions are similar as labels.
<p align="center">
  <img src="https://github.com/HazuanAiman/Diamond_price/blob/main/images/diamondsprice.png">
<p>
