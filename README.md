# Twitter Disaster Tweet Classifier

**Description:**

Twitter has become a crucial communication channel during times of emergency, enabling people to announce real-time emergencies they are witnessing. As a result, disaster relief organizations and news agencies are increasingly interested in programmatically monitoring Twitter for potential disaster announcements.

However, determining whether a tweet genuinely announces a disaster or uses disaster-related words metaphorically is not always straightforward. Human readers can often discern the intended meaning, but it can be challenging for machines.

In this project, I developed a machine learning model to predict which tweets are about real disasters and which ones are not. Leveraging a dataset of 10,000 tweets that were hand-classified, my approach involved using word embeddings to capture the semantic meaning of words.

**Objective:**

The primary goal of this project was to build a machine learning model capable of accurately classifying tweets as either real disaster tweets or non-disaster tweets. This model can help automate the process of identifying relevant tweets during emergencies, enabling timely response and support from disaster relief agencies.

**Data and Word Embeddings:**

The dataset provided for this project contained 10,000 tweets, each labeled as either a real disaster tweet or a non-disaster tweet. To prepare the data for modeling, I performed essential pre-processing steps such as tokenization, stop word removal, and special character handling.

I used word embeddings. That capture the semantic meaning of words by mapping them to continuous vector spaces. This allows the model to understand the context and relationships between words, potentially leading to more accurate classifications.

**Machine Learning Models and Learning Curve:**

As this was my first NLP project, my primary challenge was to learn and identify the best approach to achieve the best results. I experimented with several machine learning algorithms, including Logistic Regression, SVM,Adaboost,KNN etc. Additionally, I explored various word embedding models to find the one that best suited the task.

**Result:**

After implementing the Twitter Disaster Tweet Classifier using the Logistic Regression model, I achieved an accuracy of 0.75 on the validation set. While this accuracy is promising, I am determined to continue refining the model to further improve its performance and aim for even better accuracy

**Conclusion:**
I had a fantastic experience working on the Twitter Disaster Tweet Classifier, my first NLP project. Exploring word embeddings and machine learning techniques was both challenging and rewarding. I look forward to continuing my NLP journey, refining the model, and working on more projects to develop my skills.

**PCA result:**
![image](https://github.com/liorgabbay/tweets_desastors/assets/129048307/c38b5c9a-d1c0-4370-a97b-6183dcd5ab33)

![image](https://github.com/liorgabbay/tweets_desastors/assets/129048307/a0f49ba8-05fc-46d9-93c0-04f929cb817b)


**Models result on validation set:**

*Knn model accuracy*: 0.7519685039370079

*decision tree model accuracy*: 0.6338582677165354

*adaboost model accuracy*: 0.7047244094488189

*logistic regression model accuracy*: 0.7506561679790026

*QDA model accuracy*: 0.7349081364829396

*LDA model accuracy*: 0.7454068241469817

*neural network model accuracy*: 0.7086614173228346


























