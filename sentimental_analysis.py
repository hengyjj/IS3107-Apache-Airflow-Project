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

# Estimation Time of Completion: > 6 mins and are commented out for now.
# 2. This step will add a new column called sentiments to classify the reviews based on four scores: 
# neutrality, positivity, negativity and overall scores that descrbies the previous three scores.
sid = SentimentIntensityAnalyzer()
hotel_reviews_df_cleaned["sentiments"] = hotel_reviews_df_cleaned["review"].apply(lambda review: sid.polarity_scores(str(review)))
hotel_reviews_df_cleaned = pd.concat([hotel_reviews_df_cleaned.drop(['sentiments'], axis=1), hotel_reviews_df_cleaned['sentiments'].apply(pd.Series)], axis=1)

print("End of Task 2")

#3. This will add 2 more new columns, the number of character and number of words column based on each corresponding review
hotel_reviews_df_cleaned["num_chars"] = hotel_reviews_df_cleaned["review"].str.len()
hotel_reviews_df_cleaned["num_words"] = hotel_reviews_df_cleaned["review"].str.split().str.len()

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
tagged_documents = [TaggedDocument(str(review).split(), [i]) for i, review in enumerate(hotel_reviews_df_cleaned["cleaned_review"])]

#Train the Doc2Vec model
model = Doc2Vec(tagged_documents, vector_size=5, window=2, min_count=1, workers=4)

#Infer vectors for each document
doc2vec_df = pd.DataFrame([model.infer_vector(str(review).split()) for review in hotel_reviews_df_cleaned["cleaned_review"]])
doc2vec_df.columns = ["doc2vec_vector_" + str(i) for i in range(doc2vec_df.shape[1])]

# Concatenate the Doc2Vec vector with the original df
hotel_reviews_df_cleaned = pd.concat([hotel_reviews_df_cleaned, doc2vec_df], axis=1)

print("End of Task 4")

#5. Create TF-IDFS columns
# Create a TfidfVectorizer with a minimum document frequency of 10
tfidf = TfidfVectorizer(min_df=10)

# Fit the vectorizer to the cleaned reviews and transform the text into a matrix of TF-IDF features
hotel_reviews_df_cleaned["cleaned_review"] = hotel_reviews_df_cleaned["cleaned_review"].fillna("")
tfidf_result = tfidf.fit_transform(hotel_reviews_df_cleaned["cleaned_review"])

# Convert the result to a pandas DataFrame with the feature names as column headers
tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=tfidf.get_feature_names_out())

# Add a prefix to each column name for identification purposes
tfidf_df = tfidf_df.add_prefix('word_')

# Concatenate the original dataframe with the TF-IDF matrix
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

hotel_reviews_df_cleaned[hotel_reviews_df_cleaned["num_words"] >= 5].sort_values("pos", ascending = False)[["review", "pos"]].head(10)

print("End of Task 7")

print("End of file")