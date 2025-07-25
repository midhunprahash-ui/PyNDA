import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the dataset
df = pd.read_csv('suicide_intention_dataset.csv')

# Inspect the data
print(df.head())
print(df.info())

# Create a gender column from the one-hot encoded columns
def get_gender(row):
    if row['gender_male'] == 1:
        return 'Male'
    elif row['gender_female'] == 1:
        return 'Female'
    elif row['gender_non_binary'] == 1:
        return 'Non-binary'
    return 'Not specified'

df['gender'] = df.apply(get_gender, axis=1)


# Distribution of Intention Score
fig_intention = px.histogram(df, x='intention_score', title='Distribution of Intention Score', labels={'intention_score': 'Intention Score'})
fig_intention.write_json("intention_distribution.json")


# Distribution of Age
fig_age = px.histogram(df, x='age', title='Distribution of Age', nbins=20, labels={'age': 'Age'})
fig_age.write_json("age_distribution.json")


# Distribution of Gender
gender_counts = df['gender'].value_counts().reset_index()
gender_counts.columns = ['gender', 'count']
fig_gender = px.pie(gender_counts, values='count', names='gender', title='Distribution of Gender')
fig_gender.write_json("gender_distribution.json")

# Intention Score vs. Age
fig_scatter = px.scatter(df, x='age', y='intention_score', title='Intention Score vs. Age',
                         labels={'age': 'Age', 'intention_score': 'Intention Score'},
                         trendline='ols', trendline_color_override="red")
fig_scatter.write_json("intention_vs_age.json")

# Intention score by gender
fig_box_gender = px.box(df, x='gender', y='intention_score', title='Intention Score by Gender',
                        labels={'gender': 'Gender', 'intention_score': 'Intention Score'})
fig_box_gender.write_json("intention_by_gender.json")