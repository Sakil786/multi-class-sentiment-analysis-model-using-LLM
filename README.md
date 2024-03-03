# multi-class-sentiment-analysis-model-using-LLM
## Problem Statement
### Dataset File: Topical chat .csv
This dataset consists of over 8000 conversations and over 184000 messages within each message, there is a conversation id, which is basically which conversation the message takes place in. Each message is either the start of a conversation or a reply from the previous message. There is also a sentiment, which represents the emotion that the person who sent the message is feeling. There are 8 sentiments: Angry,Curious to Dive Deeper, Disguised, Fearful, Happy, Sad, and Surprised. Sentiment Analysis: Build a multi-class sentiment analysis model based on this dataset. Please report metrics for the model.

## SOLUTION
### DATASET
The dataset(Topical chat .csv) exhibits class imbalance with eight categories: Angry, Curious to Dive Deeper, Disguised, Fearful, Happy, Sad, and Surprised.

![Percentage Distribution of Sentiments](https://github.com/Sakil786/multi-class-sentiment-analysis-model-using-LLM/blob/main/imag1.png)

From the above figure, it becomes apparent that the classes Sad, Disguised, Fearful, and Angry have a notably low occurrence.The column ”message” has five records that are missing.
## DATA PREPORCESSING AND DATA CLEANING
The subsequent actions are taken in this step:
- Eliminating the missing records, given that there are only five instances of such records.
- Converting the text to lowercase
- Removing special characters and digits
- Removing stop words and applying lemmatization
## APPROACH
Initially, we adhere to the traditional Machine learning algorithmic approach, and subsequently, we employ deep learning techniques.
### Machine Learning Approach
In this approach , we used two algorithms:
- Naive Bayes Classifier
- Randomforest Classifier
#### Naive Bayes Classifier
As Naive Bayes handles both categorical target variable as well as Numerical target variable.I have followed both techniques:In first technique, I have not converted target variable into numerical variable.In the second technique, I have used label encoding to convert target variable into numerical variable.
##### Result & Metrics: 
The performance of the algorithm did not improve in both techniques.
