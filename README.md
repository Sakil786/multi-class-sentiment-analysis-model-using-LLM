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
