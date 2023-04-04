from google.cloud import bigquery
import os
import nltk 
nltk.download('popular')
nltk.download('vader_lexicon')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve


# Commands to run in terminal: 
# pip install google-cloud-bigquery
# pip install pandas 
# pip install gensim
# pip install wordcloud

# 1. Retrieving data from BigQuery (Estimation Time of Completion: 2 mins)
home_dir = os.environ['HOME']
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f'{home_dir}/is3107Project.json'

client = bigquery.Client()
table_ref = client.dataset("is3107_projectv2").table("is3107_projecv2")
rows = client.list_rows(table_ref)
data = [row for row in rows]
columns = ["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", 
                          "Hotel_Name", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", 
                          "Total_Number_of_Reviews", "Positive_Review", "Review_Total_Positive_Word_Counts", 
                          "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", 
                          "lat", "lng", "review", "is_bad_review", "cleaned_review"]

hotel_reviews_df_cleaned = pd.DataFrame.from_records(data, columns=columns)
hotel_reviews_df_cleaned.to_csv("cleaned.csv")

print("End of Task 1")

# Estimation Time of Completion: > 5 mins and are commented out for now.
# 2. This step will add a new column called sentiments to classify the reviews based on four scores: 
# neutrality, positivity, negativity and overall scores that descrbies the previous three scores.
hotel_reviews_df_cleaned = hotel_reviews_df_cleaned[["review", "is_bad_review", "cleaned_review"]]
sid = SentimentIntensityAnalyzer()
hotel_reviews_df_cleaned["sentiments"] = hotel_reviews_df_cleaned["review"].apply(lambda review: sid.polarity_scores(str(review)))
hotel_reviews_df_cleaned = pd.concat([hotel_reviews_df_cleaned.drop(['sentiments'], axis=1), hotel_reviews_df_cleaned['sentiments'].apply(pd.Series)], axis=1)

print("End of Task 2")

#3. This will add 2 more new columns, the number of character and number of words column based on each corresponding review
hotel_reviews_df_cleaned["num_chars"] = hotel_reviews_df_cleaned["review"].apply(lambda x: len(str(x)))
hotel_reviews_df_cleaned["num_words"] = hotel_reviews_df_cleaned["review"].apply(lambda x: len(str(x).split(" ")))


print("End of Task 3")

#4. Create doc2vec vector columns
# It is using the gensim library to create a Doc2Vec model and apply it to the cleaned review texts, 
# then concatenating the resulting vectors with the original DataFrame to create new columns.
# Doc2Vec is an unsupervised machine learning algorithm that learns fixed-length vector representations 
# (embeddings) from variable-length pieces of texts, such as documents, paragraphs, or sentences. 
# These embeddings can be used for tasks like text classification, clustering, and similarity matching. 
# Doc2Vec is an extension of Word2Vec, which learns embeddings for individual words. 
# Unlike Word2Vec, Doc2Vec learns a separate embedding for each document, while still taking into account 
# the words in the document.

# Create tagged documents
tagged_documents = [TaggedDocument(str(review).split(" "), [i]) for i, review in enumerate(hotel_reviews_df_cleaned["cleaned_review"])]

#Train the Doc2Vec model
model = Doc2Vec(tagged_documents, vector_size=5, window=2, min_count=1, workers=4)

#Infer vectors for each document
doc2vec_df = pd.DataFrame([model.infer_vector(str(review).split(" ")) for review in hotel_reviews_df_cleaned["cleaned_review"]])
doc2vec_df.columns = ["doc2vec_vector_" + str(i) for i in range(doc2vec_df.shape[1])]

# Concatenate the Doc2Vec vector with the original df
hotel_reviews_df_cleaned = pd.concat([hotel_reviews_df_cleaned, doc2vec_df], axis=1)

print("End of Task 4")

#5. Create TF-IDFS columns (sklearn.__version__ >= 1.0.x by "pip install --upgrade scikit-learn")
# Create a TfidfVectorizer with a minimum document frequency of 10
tfidf = TfidfVectorizer(min_df=10)

# # Fit the vectorizer to the cleaned reviews and transform the text into a matrix of TF-IDF features
hotel_reviews_df_cleaned["cleaned_review"] = hotel_reviews_df_cleaned["cleaned_review"].fillna("")
tfidf_result = tfidf.fit_transform(hotel_reviews_df_cleaned["cleaned_review"])


# # Convert the result to a pandas DataFrame with the feature names as column headers
tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=tfidf.get_feature_names_out())

# # Add a prefix to each column name for identification purposes
tfidf_df = tfidf_df.add_prefix('word_')

# # Concatenate the original dataframe with the TF-IDF matrix
hotel_reviews_df_cleaned = pd.concat([hotel_reviews_df_cleaned, tfidf_df], axis=1)

print("End of Task 5")

# 6. Interested to find out the percentage of the dataset that is considered a bad review and good review
# This will help us see whether the dataset is balanced or imbalance, and further understand the skewness
total_Bad_Reviews = "is_bad_review"
results = hotel_reviews_df_cleaned[total_Bad_Reviews].value_counts()

#Get total reviews in the data set
#Query the bad reviews / total reviews * 100% to get percentage of bad reviews same for good reviews
totalReviews = results[0] + results[1]
goodReview = results[0]
badReview = results[1]
numOfBadReviews = badReview / totalReviews
print(round(numOfBadReviews, 3) * 100)

numOfGoodReviews = goodReview / totalReviews
print(round(numOfGoodReviews, 3) * 100)

print(results)

#From this we can see that the only 4.3% of the reviews given are bad and 95.7% are good.
#Dataset is not balanced but also can be used an indicator for client to know that they are doing a good job

print("End of Task 6")

#7. Interested to find out the most used words in the reviews, regardless of good or bad
#This helps the client to see what is the sentiment about the hotel among previous guest
#Examples are "Expensive" which could indicate the per night prices are too high and or
#Small, which could indicate the rooms are too small. 
#Further investigation would be needed
def generateWordCloud(data, title = None):
    
    interestedData = str(data)
    
    wordCloud = WordCloud(
        background_color = 'white',
        max_words = 400,
        max_font_size = 40, 
        scale = 3, 
        random_state = 42
    ).generate(interestedData)

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')

    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordCloud)
    plt.show()


criteria = "review"
allHotelData = hotel_reviews_df_cleaned[criteria]
generateWordCloud(allHotelData)

#7. Get the first 10 highest reviews with a positive Sentiment
totalNumOfWords = hotel_reviews_df_cleaned["num_words"] 
totalNumOfWordsAboveFive = totalNumOfWords >= 5

getTotalNumOfWordsAboveFive = hotel_reviews_df_cleaned[totalNumOfWordsAboveFive]
getSortedPositiveValue = getTotalNumOfWordsAboveFive.sort_values("pos", ascending = False)[["review", "pos"]].head(10)
print(getSortedPositiveValue)

print("End of Task 7")

#8. Get the first 10 highest reviews with a Negative Sentiment
totalNumOfWords = hotel_reviews_df_cleaned["num_words"]
totalNumOfWordsAboveFive = totalNumOfWords >= 5

getTotalNumOfWordsAboveFive = hotel_reviews_df_cleaned[totalNumOfWordsAboveFive]
getSortedPositiveValue = getTotalNumOfWordsAboveFive.sort_values("neg", ascending = False)[["review", "neg"]].head(10)
print(getSortedPositiveValue)

# To show good reviews only
for label, is_good_review in [("Good reviews", 0)]:
    reviewToPlot = is_good_review
    goodReview = hotel_reviews_df_cleaned['is_bad_review'] == reviewToPlot
    group = hotel_reviews_df_cleaned[goodReview]
    
    sns.displot(group['compound'], label = label, kind = "kde")
    sns.histplot(group['compound'], label = label, kde = True, color = "green", fill = False)
print("End of Task 8")

# To show Bad reviews only
for label, is_bad_review in [("Bad reviews", 1)]:
    reviewToPlot = is_bad_review
    badReview = hotel_reviews_df_cleaned['is_bad_review'] == reviewToPlot
    group = hotel_reviews_df_cleaned[badReview]
    
    sns.displot(group['compound'], label = label, kind = "kde")
    sns.histplot(group['compound'], label = label, kde = True, color = "red", fill = False)
print("End of Task 9")

#Plot sentiment distribution for positive and negative reviews 
for label, is_bad_review in [("Good reviews", 0), ("Bad reviews", 1)]:
    reviewToPlot = is_bad_review
    badReview = hotel_reviews_df_cleaned['is_bad_review'] == reviewToPlot
    group = hotel_reviews_df_cleaned[badReview]
    
    sns.histplot(group['compound'], kde = True, label = label, color = "blue", fill = False)
    sns.displot(group['compound'], label = label, kind = "kde", color = "red", fill = False)
    
for label, is_bad_review in [("Good reviews", 0)]:
    reviewToPlot = is_bad_review
    badReview = hotel_reviews_df_cleaned['is_bad_review'] == reviewToPlot
    group = hotel_reviews_df_cleaned[badReview]
    
sns.distplot(group['compound'], hist = False, label = label,  color = "green")
print("End of Task 10")

# Modeling all Reviewers Score

# Feature selection
label = "is_bad_review"
ignore = [label, "review", "cleaned_review"]

trainingData = [i for i in hotel_reviews_df_cleaned.columns if i not in ignore]

# split the data into train and test
trainX, testX, trainY, testY = train_test_split(hotel_reviews_df_cleaned[trainingData],
                                                    hotel_reviews_df_cleaned[label], 
                                                    test_size = 0.20, random_state = 42)
print("End of Task 11")

#Use a random forest classifier to train the model
rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)
rfc.fit(trainX, trainY)

#12. Show importance
importances = pd.DataFrame({"Training Data": trainingData, "Importance Score": rfc.feature_importances_}).sort_values("Importance Score", ascending = False)
importances.head(30)
print("End of Task 12")

#13. ROC Curve

probY = rfc.predict_proba(testX)
predY = probY[:, 1]
falsePositiveRate, truePositiveRate, thresholds = roc_curve(testY, predY, pos_label = 1)

rocAreaUnderCurve = roc_auc_score(testY, predY)

plt.figure(figsize=(15, 10))
lw = 2
plt.plot(falsePositiveRate, 
         truePositiveRate, 
         color = 'red',
         lw = lw, 
         label ='ROC curve (area = %0.2f)' % rocAreaUnderCurve)

plt.plot([0, 1], 
         [0, 1], 
         lw = lw, 
         linestyle ='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Graph')
plt.legend(loc = "upper left")
plt.show()
print("End of Task 13")

#Draw the Precision Recall Curve

averagePrecision = average_precision_score(testY, predY)

precision, recall, _ = precision_recall_curve(testY, predY)

plt.figure(1, 
           figsize=(15, 10))

plt.step(recall, 
         precision, 
         color ='black', 
         alpha = 0.2, 
         where ='post')

plt.fill_between(recall, 
                 precision, 
                 alpha = 0.2, 
                 color ='red', 
                 step ='post')

plt.xlabel('Recall Rate')
plt.ylabel('Precision Rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(averagePrecision))
plt.show()
print("End of Task 13")

print("End of file")