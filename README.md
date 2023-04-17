# IS3107_Project
Hotel Review Analysis

## Folder/File Structure
1. Airflow File: airflow_cleaning.py
2. Machine Learning: sentimental_analysis.py
3. Data Visualisation: data_visualisation_cleaned_with_tags.ipynb
4. Google Lookers Link: google_lookers_dashboard_link.txt
5. Requirements File: requirements.txt
6. Archived folder contains files that are either duplicates or incomplete codes of the final submission files.

## Steps to get airflow pipeline working
1. create new virtual env
2. go inside virtual env
3. install airflow using the command provided in "airflow_installation_commands.txt"
4. then install required packages for our project running "pip install -r requirements.txt"
5. start airflow server by running "airflow webserver --port 8081 -D
6. then run command "airflow scheduler"
7. run your airflow DAG task.
