#Import library and packagess
import pandas as pd
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

#Load the data
hotel_reviews_df = pd.read_csv("Hotel_Reviews.csv")
#print(hotel_reviews_df.head())

#Append the positive and negative reviews
hotel_reviews_df["review"] = hotel_reviews_df["Negative_Review"] + hotel_reviews_df["Positive_Review"]

#Assign the label to the newly created positive and negative reviews (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html)
def make_review(row):
    if row["Reviewer_Score"] < 5:
        return 1
    else:
        return 0

hotel_reviews_df["is_bad_review"] = hotel_reviews_df.apply(make_review, axis=1)
  


#Speed up computations
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
hotel_reviews_df = hotel_reviews_df.sample(frac = 0.1, replace = False, random_state=42)

def remove_review_column(row):
    if row["Negative_Review"] == "No Negative":
        return ""
    if row["Positive_Review"] == "No Positive":
        return ""
    
hotel_reviews_df["review"] = hotel_reviews_df.apply(remove_review_column, axis=1)
print(hotel_reviews_df.head())   

hotel_reviews_df.to_csv('hotel_reviews_df_2.csv',index=False)