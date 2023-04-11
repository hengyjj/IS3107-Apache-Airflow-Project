import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import time

df_uncleaned = pd.read_csv("/work/Hotel_Reviews.csv");
df = pd.read_csv("/work/Hotel_Reviews.csv");

# Bar chart using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='Hotel_Name', y='Average_Score', data=df)
plt.xticks(rotation=90)
plt.xlabel('Hotel Name')
plt.ylabel('Average Score')
plt.title('Average Score of Each Hotel')
plt.show()

# Stacked bar chart using Seaborn
df['Positive_Review_Length'] = df['Positive_Review'].apply(lambda x: len(x.split()))
df['Negative_Review_Length'] = df['Negative_Review'].apply(lambda x: len(x.split()))
df_stacked = df.groupby('Hotel_Name')[['Positive_Review_Length', 'Negative_Review_Length']].sum().reset_index()
df_stacked = df_stacked.melt(id_vars='Hotel_Name', var_name='Review Type', value_name='Word Count')
sns.barplot(x='Hotel_Name', y='Word Count', hue='Review Type', data=df_stacked)
plt.xticks(rotation=90)
plt.xlabel('Hotel Name')
plt.ylabel('Word Count')
plt.title('Proportion of Negative and Positive Reviews for Each Hotel')
plt.show()

# Waterfall chart
plt.figure(figsize=(10,8))
sns.barplot(x='Reviewer_Score', y='Total_Number_of_Reviews_Reviewer_Has_Given', data=df)
plt.xticks(rotation=90)
plt.show()

# Bubble chart
plt.figure(figsize=(10,8))
sns.scatterplot(x='Reviewer_Score', y='Average_Score', size='Total_Number_of_Reviews', data=df)
plt.show()

# Sunburst chart
plt.figure(figsize=(10,8))
df['Hotel_Name'].value_counts().plot.pie(subplots=True, wedgeprops=dict(width=0.5), textprops=dict(size=10), startangle=90, counterclock=False)
plt.show()

from wordcloud import WordCloud
df = pd.read_csv("/work/Hotel_Reviews.csv");
neg_words = ' '.join(df['Negative_Review'].values.tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neg_words)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

df = pd.read_csv("/work/Hotel_Reviews.csv");
pos_word = ' '.join(df['Positive_Review'].values.tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pos_word)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


avg_scores = df.groupby('Reviewer_Score')['Average_Score'].mean()

# Create the line chart
plt.plot(avg_scores.index, avg_scores.values)
plt.xlabel('Reviewer Score')
plt.ylabel('Average Score')
plt.title('Average Score by Reviewer Score')
plt.show()

# Create the histogram
plt.hist(df['Average_Score'], bins=20, color='blue')
plt.xlabel('Average Score')
plt.ylabel('Count')
plt.title('Distribution of Average Scores')
plt.show()

# Get the top 10 reviewed hotels by count of total reviews
top_hotels = df['Hotel_Name'].value_counts().nlargest(10)

# Print the top 10 hotels and their review counts
print('Top 10 Reviewed Hotels:')
print(top_hotels)

# Create a column with the rounded reviews
df["Reviewer_Score_Round"] = df["Reviewer_Score"].apply(lambda x: int(round(x)))

# Get the number of reviews with which scores
reviews_dist = df["Reviewer_Score_Round"].value_counts().sort_index()
bar = reviews_dist.plot.bar(figsize =(10,7))
plt.title("Distribution of reviews", fontsize = 18)
plt.axvline(df["Reviewer_Score"].mean()-2, 0 ,1, color = "grey", lw = 3)
plt.text(6, -15000, "average", fontsize = 14, color = "grey")
plt.ylabel("Count", fontsize = 18)
bar.tick_params(labelsize=16)

# Remove the column "Reviewer_Score_Round"
df.drop("Reviewer_Score_Round", axis = 1, inplace = True)

# Correlation
df_corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(df_corr, annot = True)
plt.title("Correlation between the variables", fontsize = 22)
plt.show()

# Compute the difference between the highest and lowest average score for each hotel
hotel_diff = df.groupby('Hotel_Name')['Average_Score'].apply(lambda x: x.max() - x.min())

# Get the top 10 hotels with the largest difference in average score
top_hotels = hotel_diff.nlargest(10)

# Print the top 10 hotels and their score differences
print('Top 10 Hotels with the Most Decrease in Average Score:')
print(top_hotels)