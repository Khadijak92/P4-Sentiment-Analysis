# Sentiment Analysis on Amazon Item Reviews
## Table of Contents
## Overview
- Data Collection and Storage
- Data Preprocessing
- Modeling
- Model Initialization
- Model Training
- Model Evaluation
- Model Optimization
- Model Deployment
- Analysis/Future Recommendations
- Installation
- Usage
- Contributing
- License

## Overview

This project is centered on performing sentiment analysis on Amazon item reviews, leveraging advanced machine learning and deep learning techniques to classify the sentiment of the reviews accurately. The primary tools and libraries used in this project include TensorFlow, Keras, pandas, matplotlib, and NLTK. The ultimate objective is to develop a predictive model that can achieve a classification accuracy of at least 75% and an R-squared value of 0.80 or higher.

## Data Collection and Storage

The data for this project is sourced from Amazon item reviews. The reviews are stored in a SQL database and retrieved using SQL queries. The data includes review text and corresponding sentiment labels.

## Data Preprocessing
To ensure the model performs well, the data is cleaned, normalized, and standardized. The preprocessing steps include:

Cleaning: Removing HTML tags, special characters, and stopwords using NLTK.
Normalization: Converting text to lowercase.
Tokenization: Splitting text into individual words.
Lemmatization: Reducing words to their base form.
Standardization: Transforming data into a standard format.

## Modeling
## Model Initialization
The model is initialized using TensorFlow and Keras. The architecture includes:

An embedding layer for word embeddings.
LSTM layers for capturing sequential dependencies.
Dense layers for classification.
## Model Training
The model is trained on the preprocessed data using TensorFlow and Keras. The training process involves:

Splitting the data into training and validation sets.
Compiling the model with appropriate loss function and optimizer.
Fitting the model on the training data with validation.
## Model Evaluation
The model's performance is evaluated using accuracy and R-squared metrics. The trained model achieves:

Classification Accuracy: 75%
R-squared: 0.80
## Model Optimization
The optimization process includes iterative changes to the model architecture, hyperparameters, and training process. The results of each iteration are documented in a CSV file. Key steps in the optimization process:

Adjusting the number of LSTM layers and units.
Tuning the dropout rate.
Experimenting with different optimizers and learning rates.
## Model Deployment
The trained model is deployed using Flask. The deployment includes:

A RESTful API that accepts review text and returns sentiment predictions.
Integration with a front-end interface for user interaction.

## Analysis/Future Recommendations
Our model aimed to evaluate different Amazon reviews of a certain product in order to learn how well the general public perceives this product (in this case, it was different Amazon products, specifically different versions of the Kindle and different versions of the Amazon Alexa). We initially cleaned up the model by removing unnecessary rows, and translating the text into numbers to turn the written reviews into something that could be analyzed my our model in a predictive manner.

From there, we padded our sequence for training and testing, which we did after normalizing the  features. After this, we initialized our model with two hidden layers, then compiled and trained our model before finally evaluating the model and plotting the accuracy results.

Unfortunately, we found that our model was very inaccurate with am accuracy rating of just 0.06, which was below our target 0.75. We took this as a sign that our model was not able to accurately predict the tone of the reviews and instead did almost the opposite. Therefore, after a lot of thought about what could be some steps we would take next time to more accurately predict the review sentiment, we would most likely use a non-categorical variable such as number of stars given for the model to predict, and try to limit the model to just predict one type of product. By doing this, the model would not get confused and compare reviews from two uncomparable products, and it would also ensure that we have a variable we can more easily predict with our model.
## Installation
To install the project dependencies, run:

bash
Copy code
pip install -r requirements.txt
Usage
To start the Flask server, run:

bash
Copy code
python app.py
Access the web interface at http://127.0.0.1:5000.
