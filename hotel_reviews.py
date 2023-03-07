#1. Import library and packagess
import pandas as pd
import string
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

#2. Load the data
hotel_reviews_df = pd.read_csv("Hotel_Reviews.csv")
#print(hotel_reviews_df.head())

#3. Append the positive and negative reviews
hotel_reviews_df["review"] = hotel_reviews_df["Negative_Review"] + hotel_reviews_df["Positive_Review"]

#4. Assign the label to the newly created positive and negative reviews (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html)
def make_review(row):
    if row["Reviewer_Score"] < 5:
        return 1
    else:
        return 0

hotel_reviews_df["is_bad_review"] = hotel_reviews_df.apply(make_review, axis=1)

#5. Speed up computations by sampling
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
hotel_reviews_df = hotel_reviews_df.sample(frac = 0.1, replace = False, random_state=42)

#6. Cleaning reviews to exclude "No Negative" & "No Positive"
hotel_reviews_df["review"] = hotel_reviews_df["review"].replace(
    {"No Negative|No Positive": ""}, regex=True)


#7. Classifying reviews based on WordNet Part-of-speech (POS) tag
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    
    else:
        return wordnet.NOUN

#8. Cleaning reviews
def clean_review(review):
    #Convert all reviews to lowercase
    review = review.lower()

    #Removing punctuation
    review = review.replace(string.punctuation, '')
    review = review.split(' ')

    #Filtering out digits
    review = [word for word in review if not any(c.isnumeric() for c in word)]

    #Removing stop words such as a, an, the, is, are, was, were, and etc.
    stop = stopwords.words('english')
    review = [word for word in review if word not in stop]

    #Removing empty words
    review = [word for word in review if len(word) > 0]

    #POS tag
    pos_tags = get_wordnet_pos(review)

    #Lemmatise words E.g. Running, Ran, Run -> Run (Base form)
    review = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    #Keeping words that has more than 1 letter
    review = [word for word in review if len(word) > 1]

    #Join all string
    review = " ".join(review)
    return (review)

#9. Downloading Popular package from NLTK
nltk.download('popular')


################## Codes For Testing ##################
# Printing first 5 rows
# print(hotel_reviews_df.head()) 

# Uncomment the below two codes to get a smaller csv file (input the column that you wish to check)
# hotel_reviews_df = hotel_reviews_df[["review", "is_bad_review"]]
# hotel_reviews_df.to_csv('hotel_reviews_df_2.csv', index=False)