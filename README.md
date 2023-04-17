# IS3107_Project AY2022/2023 Semester 2
Hotel Review Analysis

## Information
Prepared By: Group 8
Group Member: Alwin Ng Jun Wei, Heng Yi Jie, Lee Jia Wei, Lim Wee Kiat, Tan Jun Bin

## Folder/File Structure
1. Airflow File: airflow_cleaning.py
2. Machine Learning: sentimental_analysis.py
3. Data Visualisation: data_visualisation_cleaned_with_tags.ipynb
4. Google Lookers Link: google_lookers_dashboard_link.txt
5. Requirements File: requirements.txt
6. Archived folder contains files that are either duplicates or incomplete codes of the final submission files.

## Steps to get airflow pipeline working
1. Create new virtual env.
2. Go inside virtual env.
3. Install Airflow using the command provided in "airflow_installation_commands.txt"
4. Install required packages for our project running "pip install -r requirements.txt"
5. Start Airflow server by running "airflow webserver --port 8081 -D"
6. Run command "airflow scheduler"
7. Run your airflow DAG task.
