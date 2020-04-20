#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ตอนที่ 1: การทดลองเตรียมข้อมูลและแสดงรายละเอียดข้อมูลเชิงกราฟ
### Import Lib (numpy, pandas, matplotlib, sklearn, keras)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture

import re

plt.style.use('ggplot')

### โหลดข้อมูล

rating_df = pd.read_csv(root_dir + '/ratings.csv')
movie_df = pd.read_csv(root_dir + '/movies.csv')
tag_df = pd.read_csv(root_dir + '/tags.csv')

### Data Preprocessing

rating_df['uts'] = pd.to_datetime(rating_df['timestamp'], unit='s', origin=pd.Timestamp('1970-01-01'))
rating_df['year'] = rating_df['uts'].apply(lambda uts : uts.year)
rating_df

tag_df['uts'] = pd.to_datetime(tag_df['timestamp'], unit='s', origin=pd.Timestamp('1970-01-01'))
tag_df

movie_df['year'] = movie_df['title'].str.extract(r"\((\d+)\)")
movie_df['year'] .fillna(method='ffill', inplace=True)              # เติมค่า null value(year) ด้วย ปีก่อนหน้า
movie_df['year'] = movie_df['year'].astype('int32')                 # cast to integer

movie_df['year'] = movie_df.apply(lambda row: row['title'].strip()[-5:-1] if row['year'] < 1800 else row['year'], axis=1)
movie_df['year'] = movie_df['year'].astype('int32')                 # cast to integer

movie_df.describe()

# Create list of genre dataframe 
temp_genre = movie_df.copy()
temp_genre['genres_split'] = temp_genre['genres'].str.split(pat = "|")
temp_genre = temp_genre[['movieId', 'genres_split']]
temp_genre.columns = ['movieId', 'genres']
temp_genre

# Iterate each row to find unique genre
genre_set = set()
for index, row in temp_genre.iterrows():
  [genre_set.add(g) for g in row['genres']]
  
# Create genre dataframe 
genre_df = pd.DataFrame(index=temp_genre.index, columns=genre_set)
genre_df['movieId']  = temp_genre['movieId']

for index, row in temp_genre.iterrows():
  for genre in row['genres']:
    genre_df.loc[index, genre] = 1 

genre_df.fillna(0, inplace=True)
new_column = ['movieId'] + (genre_df.columns[: -1].tolist())
genre_df = genre_df[new_column]
genre_df.index.name = 'No.'
genre_df

### Data Visualization

#  จำนวน released movies ในแต่ละปี
release_movie = movie_df[['year', 'movieId']].groupby(['year']).count()
release_movie.columns = ['Number of movies']
release_movie.plot.bar(figsize=(30,10), title='Number of movies in each year')

# แสดงกราฟค่า จำนวนการให้rating ในแต่ละปี
rate_movie = rating_df[['year', 'rating']].groupby(['year']).count()
rate_movie.columns = ['Number of rating']
rate_movie.plot.bar(figsize=(30,10), title='Number of ratings in each year')

# แสดงกราฟค่า จำนวน movies ในแต่ละ genre
movie_genre = pd.DataFrame(genre_df.iloc[:, 1:].sum())
movie_genre.columns = ['Number of movies']
movie_genre.plot.bar(figsize=(30,10), title='Number of movies in each genre')

# # Create list of genre dataframe 
# temp_genre2 = movie_df.copy()
# temp_genre2['genres_split'] = temp_genre2['genres'].str.split(pat = "|")
# temp_genre2 = temp_genre2[['movieId', 'genres_split', 'year']]
# temp_genre2.columns = ['movieId', 'genres', 'year']
# temp_genre2

# # Create dataframe with 'movieId', 'genres', 'year'
# genre_mv_yr = pd.DataFrame(columns=['movieId', 'genres', 'year'])
# count_row = 0

# for index, row in temp_genre2.iterrows():
#   genre_set = list()
#   [genre_set.append(g) for g in row['genres']]
  
#   for i in genre_set:
#     genre_mv_yr.loc[count_row] = [row['movieId'], i, row['year']]
#     count_row += 1
# genre_mv_yr.to_csv(root_dir + '/genre_mv_yr.csv')
# genre_mv_yr

# =====================================================================

genre_mv_yr = pd.read_csv(root_dir + '/genre_mv_yr.csv')
genre_mv_yr.drop('Unnamed: 0', axis=1, inplace=True)
genre_mv_yr


# แสดงกราฟ (y-axis: stacked graph) ค่าจำนวน movie แต่ละ genre ในแต่ละปี (x-axis)
genre_mv_yr_plot = genre_mv_yr.groupby(['genres', 'year']).count()
genre_mv_yr_plot

genre_mv_yr_plot.unstack(level=0).plot.area(stacked=True, figsize=(40,10), title='Number of movies in each genre in each year')

#  แสดงกราฟ Histogram ของการกระจายของค่าเฉลี่ย movie rating ใน dataset
sample_rating_df = rating_df[['userId', 'movieId', 'rating', ]].iloc[:100000]
sample_rating_df = sample_rating_df.sort_values(by=['userId', 'movieId'])
sample_rating_df

user_rate_mt = pd.DataFrame(index=sample_rating_df['userId'].unique(), columns=sample_rating_df['movieId'].sort_values().unique())      # Nan matrix

for index, row in sample_rating_df.iterrows():
    user_rate_mt.loc[row['userId'], row['movieId']] = row['rating']
  
user_rate_mt

user_rate_mean = user_rate_mt.mean(axis=0)
user_rate_mean_df = pd.DataFrame(user_rate_mean)
user_rate_mean_df.columns = ['movie_rating_mean']
user_rate_mean_df.describe()

user_rate_mean_df.hist(bins=10, figsize=(30,10))

