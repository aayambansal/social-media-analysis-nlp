Mental Health Crisis Detection through Social Media Analysis
Overview
This project focuses on the early detection of mental health crises by analyzing linguistic patterns in social media posts, particularly from Twitter. Using a combination of Natural Language Processing (NLP) and Deep Learning techniques, the project aims to identify subtle language cues that may indicate a deterioration in mental health. The model leverages BERT and LSTM architectures to predict changes in users' mental states based on their tweets.

Key Features:
Data Collection: Collect public tweets using the Twitter API.
Natural Language Processing: Preprocess the data by cleaning, tokenizing, and extracting important linguistic features.
Hybrid Model: Use a BERT-LSTM deep learning architecture to classify users based on their mental health.
Ethics: Handle sensitive data responsibly, ensuring anonymity and compliance with ethical guidelines.
Table of Contents
Overview
Dataset
Model
Setup
Usage
Results
Ethics
Future Directions
License
Dataset
Tweets are collected using the Twitter API for two groups of users:

Users who self-identify as having a mental health condition.
A control group of users with no mental health self-identification.
The data is preprocessed to remove personally identifiable information (PII), URLs, and irrelevant tokens. The final dataset is stored in CSV format for further analysis.

Data Collection Program:
You can collect the dataset using the provided Python script in the dataset_collection.py file. This script uses the Tweepy library to collect tweets from user timelines based on user IDs.

Model
This project uses a hybrid BERT-LSTM model for detecting early signs of mental health crises:

BERT: Pre-trained language model to capture contextual embeddings from user tweets.
LSTM: Sequential model that processes these embeddings to identify patterns over time.
The model is trained using PyTorch with metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
Setup
Prerequisites:
Python 3.8+
Twitter Developer Account (for API access)
Required Python Libraries:
tweepy
torch
transformers
pandas
sklearn
nltk
Install the required libraries:

pip install tweepy torch transformers pandas scikit-learn nltk

Setup Twitter API:
Create a Twitter Developer account and generate API keys (API Key, API Secret, Access Token, Access Secret).
Set up these credentials in the dataset_collection.py script to collect data.
Usage
1. Data Collection:
Use the dataset_collection.py script to collect tweets from specified users:

python dataset_collection.py

The collected tweets will be saved in mental_health_tweets.csv file.

2. Data Preprocessing:
Preprocess the dataset by tokenizing and extracting relevant linguistic features using libraries like NLTK and Hugging Face Transformers.

3. Model Training:
Train the BERT-LSTM model on the preprocessed data using the train_model.py script:

python train_model.py

Adjust parameters like learning rate, batch size, and number of epochs in the script.

Results
Accuracy: 83% for early detection of mental health crises.
Key Linguistic Features: Use of first-person pronouns, negative emotion words, sentence complexity reduction, and late-night posting patterns were the most significant indicators.
The model demonstrated the ability to predict mental health declines on average 6.3 weeks before explicit mentions of crises in posts.
Ethics
This project adheres to the following ethical guidelines:

Privacy: All collected data is anonymized, and no attempts are made to re-identify users.
Consent: The project uses only publicly available data from Twitter.
Avoiding Stigmatization: The project ensures that results are interpreted with care, avoiding inappropriate labeling or stigmatization of individuals.
Data Security: Ensures that sensitive data is securely handled and stored.
Future Directions
Multimodal Analysis: Combine text analysis with image and video analysis to provide a more comprehensive view of mental health indicators.
Cross-platform Analysis: Extend the project to other social media platforms like Reddit and Facebook.
Explainable AI: Incorporate techniques like SHAP for better interpretability of the model's decisions.


