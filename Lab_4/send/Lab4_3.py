#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ตอนที่ 3: การทดลองสร้างระบบแนะนำสินค้า (Recommendation system) จากข้อมูล movie_matrix
### คำนวณความคล้ายกันของ movie genre ของคู่ ‘movieId’ ใดๆ จากตาราง movie genre feature 
# Random 30 movie
random_genre_df =  genre_df.sample(30)
random_genre_df.reset_index(inplace=True, drop=True)
random_genre_df.set_index('movieId', inplace=True)
random_genre_df.head()

# cosine_similarity
cos_genre_sim = cosine_similarity(random_genre_df, random_genre_df)
cos_genre_sim = pd.DataFrame(cos_genre_sim)
cos_genre_sim.columns.name = 'userid'
cos_genre_sim.index.name = 'userid'

fig, ax = plt.subplots(figsize=(30,30))  
sns.heatmap(cos_genre_sim, annot = True, vmin=-1, vmax=1, center= 0, linewidths=.5, ax=ax, xticklabels=random_mt.index, yticklabels=random_mt.index)

#  Pearson’s similarity()
pearson_genre_sim = random_genre_df.T.corr( method ='pearson' )
fig, ax = plt.subplots(figsize=(30,30))  
sns.heatmap(pearson_genre_sim, annot = True, vmin=-1, vmax=1, center= 0, linewidths=.5, ax=ax, xticklabels=random_mt.index, yticklabels=random_mt.index)

### แสดงรูปภาพ

# ตาราง user ที่มีความชอบตรงกันข้ามกันที่สุด 5 อันดับ Pearson’s similarity
pear_sim_genre, pear_opp_genre = display_top_sim(pearson_genre_sim, 'Pearson’s similarity')

# รายการของ user ที่ให้ rating >= 3.0 ซึ่งสามารถแนะนำ movie ในรายการข้อ 3.2.1 ให้ได้
pear_sim_genre_df = pd.DataFrame(pear_sim_genre, columns=['similarity'])

# get movie id
sim_movies = set()
for t in list(pear_sim_genre_df.index) :
  sim_movies.add(t[0])
  sim_movies.add(t[1])
sim_movies = list(sim_movies)

pear_sim_genre_df

user_rate3_df = user_rate_mt.loc[:, sim_movies]
user_rate3_df = user_rate3_df[user_rate3_df.gt(2.9)]
user_rate3_df.dropna(axis=0, how='all', inplace=True)     # drop row
user_rate3_df.dropna(axis=1, how='all', inplace=True)     # drop column
user_rate3_df

reccomand_movie_col = list()
for idx, row in user_rate3_df.iterrows() :
  rec_movie = list()
  for col in user_rate3_df.columns:

    if pd.notnull(row[col]):

      for mID in pear_sim_genre_df.index:
        if col in mID :
          rec_movie.append(mID[0]) if mID.index(col) else rec_movie.append(mID[1])
  reccomand_movie_col.append(set(rec_movie))

df = pd.DataFrame(pd.Series(reccomand_movie_col), columns=['reccomand_movie'])
user_rate3_df['reccomand_movie'] = df['reccomand_movie'].values
user_rate3_df.fillna('-', inplace=True)
user_rate3_df

