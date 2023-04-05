from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import bigquery
from datetime import datetime
import pandas as pd
from zipfile import ZipFile
import json
import string
import nltk
import os
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

default_args = {
    'owner': 'airflow',
}

with DAG(
    'is3107_project',
    default_args=default_args,
    description='IS3107 Project',
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    
    

    # ---------------------------- Start of Extract -----------------------------
    # This command will download data via api and download data as zip file locally at the desired path. (Requires Kaggle API key to perform. )
    # Get your home directory inside your OS, then store whatever files we got in this directory. This is to prevent hardcoding file directory
    home_dir = os.environ['HOME']

     # Load the service account credentials from the JSON key file
    my_credentials = service_account.Credentials.from_service_account_file(
        f'{home_dir}/is3107Project.json'
    )

    download_dataset = BashOperator(
        task_id='download_dataset',
        bash_command='kaggle datasets download jiashenliu/515k-hotel-reviews-data-in-europe '
        f'--path {home_dir}/',
        dag=dag
    )

    def extract(**kwargs):
        ti = kwargs['ti']
        # HAS ERROR as file is not unzipping
        with ZipFile(f'{home_dir}/515k-hotel-reviews-data-in-europe.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{home_dir}')

        print("Finish downloading and unzipping")

# ---------------------------- End of Extract -----------------------------

# ---------------------------- Start of Transform -----------------------------
    def transform(**kwargs):
        ti = kwargs['ti']
        hotel_reviews_df = pd.read_csv(f'{home_dir}/Hotel_Reviews.csv')

        # 1. Append the positive and negative reviews
        hotel_reviews_df["review"] = hotel_reviews_df["Negative_Review"] + \
            hotel_reviews_df["Positive_Review"]

        # 2. Assign the label to the newly created positive and negative reviews (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html)
        def make_review(row):
            if row["Reviewer_Score"] < 5:
                return 1
            else:
                return 0

        hotel_reviews_df["is_bad_review"] = hotel_reviews_df.apply(
            make_review, axis=1)

        # 3. Speed up computations by sampling
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
        hotel_reviews_df = hotel_reviews_df.sample(
            frac=0.1, replace=False, random_state=42)

        # 4. Cleaning reviews to exclude "No Negative" & "No Positive"
        hotel_reviews_df["review"] = hotel_reviews_df["review"].replace(
            {"No Negative|No Positive": ""}, regex=True)

        # 5. Classifying reviews based on WordNet Part-of-speech (POS) tag
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

        # 6. Cleaning reviews
        def clean_review(review):
            # Convert all reviews to lowercase
            review = review.lower()

            # Removing punctuation
            review = review.replace(string.punctuation, '')
            review = review.split(' ')

            # Filtering out digits
            review = [word for word in review if not any(
                c.isnumeric() for c in word)]

            # Removing stop words such as a, an, the, is, are, was, were, and etc.
            stop = stopwords.words('english')
            review = [word for word in review if word not in stop]

            # Removing empty words
            review = [word for word in review if len(word) > 0]

            # POS tag
            pos_tags = pos_tag(review)

            # Lemmatise words E.g. Running, Ran, Run -> Run (Base form)
            review = [WordNetLemmatizer().lemmatize(
                t[0], get_wordnet_pos(t[1])) for t in pos_tags]

            # Keeping words that has more than 1 letter
            review = [word for word in review if len(word) > 1]

            # Join all string
            review = " ".join(review)
            return (review)

        # 7. Downloading Popular package from NLTK
        nltk.download('popular')

        # 8. Clean reviews 
        hotel_reviews_df["cleaned_review"] = hotel_reviews_df["review"].apply(
            lambda x: clean_review(x))
        
        # Add a row ID column
        # IMPORTANT, MUST DO THIS, if not when we retrieved data from bigquery, we can't get the same ordering, then the word cloud will be different
        hotel_reviews_df.reset_index(drop=True, inplace=True)
        hotel_reviews_df['row_id'] = hotel_reviews_df.index + 1
        
        # Write the cleaned data to a new CSV file
        hotel_reviews_df.to_csv(
            f'{home_dir}/Hotel_Reviews_Cleaned.csv', index=False)

        print("Finish transforming")
# ---------------------------- End of Transform -----------------------------

    
# ---------------------------- Start of upload to cloud storage -----------------------------

    def uploadToCloudStorage(**kwargs):
        # Connect to Google Cloud Storage
        storage_client = storage.Client(credentials=my_credentials)
        # Get the bucket
        bucket = storage_client.get_bucket('is3107-project-group8')

        # Upload the CSV file
        blob = bucket.blob('Hotel_Reviews_Cleaned.csv')
        blob.upload_from_filename(f'{home_dir}/Hotel_Reviews_Cleaned.csv')

# ---------------------------- End of upload to cloud storage -----------------------------


# ---------------------------- Start of Transfer to Big Query -----------------------------
    def transferUploadedCSVToBigQuery(**kwargs):
        # Connect to Google Cloud BigQuery
        bigquery_client = bigquery.Client(credentials=my_credentials)
        # Get the dataset
        dataset = bigquery_client.dataset('is3107_projectv2')
        # Create the table
        table = dataset.table('is3107_projecv2')

        # Load the CSV file into the table
        job_config = bigquery.LoadJobConfig(
            autodetect=True, source_format=bigquery.SourceFormat.CSV, write_disposition='WRITE_TRUNCATE'
        )
        job = bigquery_client.load_table_from_uri('gs://is3107-project-group8/Hotel_Reviews_Cleaned.csv', table, job_config=job_config)
        job.result()
        destination_table = bigquery_client.get_table(table)  # Make an API request.
        print("Loaded {} rows into bigquery from uploaded cleaned csv data file in cloud storage.".format(destination_table.num_rows))

# ---------------------------- End of Transfer to Big Query -----------------------------
      

# ---------------------------- Start of TASKS -----------------------------

    extract_task = PythonOperator(
        task_id='extract',
        python_callable=extract,
    )  # this is the function name that this task is testing.

    transform_task = PythonOperator(
        task_id='transform',
        python_callable=transform,
    )


    uploadToCloudStorageTask = PythonOperator(
        task_id='uploadToCloudStorage',
        python_callable=uploadToCloudStorage,
    )

    transferUploadedCSVToBigQueryTask = PythonOperator(
        task_id='transferUploadedCSVToBigQuery',
        python_callable=transferUploadedCSVToBigQuery,
    )

    # ---------------------------- END of TASKS -----------------------------


    # Finaly, execute the tasks one by one
    download_dataset >> extract_task >> transform_task >> uploadToCloudStorageTask >> transferUploadedCSVToBigQueryTask
