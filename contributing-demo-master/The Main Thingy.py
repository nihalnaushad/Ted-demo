# Importing modules
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
reviews = pd.read_csv("./data/google-play-store/googleplaystore_user_reviews.csv")

# Filtering Data
# 'review' { "Variable holding DataFrame grouped by App" }
# 'review_total' { "Variable holding filtered DataFrame" }
review = reviews.groupby('App')
review_total = review.sum()

del review_total['Sentiment_Subjectivity']

review_total = review_total.sort_values('Sentiment_Polarity', ascending=False)
review_total = review_total[(review_total['Sentiment_Polarity'] > 20) | (review_total['Sentiment_Polarity'].isnull())]

# Plotting a bar graph with the Data
my_plot = review_total.plot(kind='bar',legend=None,title="Google Play Store App Review")
my_plot.set_xlabel("App Name")
my_plot.set_ylabel("Review")
plt.show()
