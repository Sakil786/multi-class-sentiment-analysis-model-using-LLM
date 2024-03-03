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
![Percentage Distribution of Sentiments](https://github.com/Sakil786/multi-class-sentiment-analysis-model-using-LLM/blob/main/image2.png)

![Percentage Distribution of Sentiments](https://github.com/Sakil786/multi-class-sentiment-analysis-model-using-LLM/blob/main/image3.png)

### Metrics Explanation :
##### Term Explanation
• **Precision**: The proportion of correctly identified instances among those labeled as positive. It measures how often the model is correct when it predicts a positive result.

• **Recall**: The proportion of actual positive instances that were correctly identified. It measures how often the model correctly identifies all positive cases.

•**F1-score**: The harmonic mean of precision and recall, providing a balanced measure of accuracy that considers both. Higher F1-scores indicate better overall performance.

• **Support**: The number of instances for each class in the dataset. It shows how much data was available for training and evaluation.

##### Class-Specific Metrics Explanation:
• **Angry, Disgusted, Fearful, Sad**: The model has 0.00 for precision, recall, and F1-score for these classes, indicating it’s not accurately identifying them.

• **Curious to dive deeper:** The model has a precision of 0.45, recall of 0.93, and F1-score of 0.61,suggesting it’s relatively good at identifying this class, but with some false positives.

• **Happy, Neutral, Surprised:** The model has moderate precision (0.42-0.47) but low recall (0.07-0.09) for these classes, indicating it’s missing many true instances while correctly identifying some.

##### Overall Performance:
• **Accuracy**: 0.45, indicating the model correctly classifies about 45% of instances overall.

• **Macro avg:** 0.44 for precision and recall, reflecting the average performance across classes without considering their distribution.

• **Weighted avg**: 0.35, considering the class distribution and giving more weight to classes with more instances.

##### Observations:
• The model struggles to identify certain classes (Angry, Disgusted, Fearful, Sad) accurately.

• It performs best for the ”Curious to dive deeper” class, but with some false positives.

• Overall accuracy is moderate, suggesting potential for improvement.

