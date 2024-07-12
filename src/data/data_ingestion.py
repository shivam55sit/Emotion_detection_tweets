import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

df.drop(columns='tweet_id', inplace=True)

final_df = df[df['sentiment'].isin(['happiness','sadness'])]

final_df['sentiment'].replace({'happiness':1,'sadness':0},inplace=True)

data_path = os.path.join('data','interim')


train_data, test_data = train_test_split(final_df,test_size=0.2,random_state=42)

train_data.to_csv(os.path.join(data_path,'train_data.csv'))
test_data.to_csv(os.path.join(data_path,'test_data.csv'))

