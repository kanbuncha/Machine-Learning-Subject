#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ตอนที่ 2: การทดลองสร้างระบบแนะนำสินค้า (Recommendation system) จากข้อมูล user_matrix
### สร้างข้อมูลความชอบของผู้ใช้แต่ละคน (user_ matrix)
sample_rating_df = rating_df[['userId', 'movieId', 'rating', ]]
sample_rating_df = sample_rating_df

# Create nan matrix 
user_rate_mt = pd.DataFrame(index=sample_rating_df['userId'].unique(), columns=sample_rating_df['movieId'].unique())      # Nan matrix

for index, row in sample_rating_df.iterrows():
    user_rate_mt.loc[row['userId'], row['movieId']] = row['rating']
user_rate_mt.fillna(0, inplace=True)
user_rate_mt.columns.name = 'movieid'
user_rate_mt.index.name = 'userid'
user_rate_mt

##  คำนวณความคล้ายกันของความชอบดูหนังของคู่ ‘userId’ ใดๆ

# Random 30 user
random_mt = user_rate_mt.sample(30)

# cosine_similarity
cos_sim = cosine_similarity(random_mt, random_mt)
cos_sim = pd.DataFrame(cos_sim)
cos_sim.columns.name = 'userid'
cos_sim.index.name = 'userid'

fig, ax = plt.subplots(figsize=(30,30))  
sns.heatmap(cos_sim, annot = True, vmin=-1, vmax=1, center= 0, linewidths=.5, ax=ax, xticklabels=random_mt.index, yticklabels=random_mt.index)

#  Pearson’s similarity()
pearson_sim = random_mt.T.corr( method ='pearson' )
fig, ax = plt.subplots(figsize=(30,30))  
sns.heatmap(pearson_sim, annot = True, vmin=-1, vmax=1, center= 0, linewidths=.5, ax=ax, xticklabels=random_mt.index, yticklabels=random_mt.index)

### แสดงตารางรายการดังนี้


def display_top_sim(corr_df, method) :
  corr_df = corr_df
  sort_corr = (corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))            #the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
  print(f"\n*************************** {method} *************************** ")
  print("------ Top 5 Similarity ------")
  print(sort_corr[0: 5])
  # print('\n---------------------------------------------------------------------\n')
  print("------ Top 5 Diffence ------")
  print(sort_corr[-5: ])

  return sort_corr[0: 5], sort_corr[-5: ]

# ตาราง user ที่มีความชอบตรงกันข้ามกันที่สุด 5 อันดับ cosine_similarity
cos_like, cos_dislike = display_top_sim(cos_sim, 'Cosine similarity')

# ตาราง user ที่มีความชอบตรงกันข้ามกันที่สุด 5 อันดับ Pearson’s similarity
pear_like, pear_dislike = display_top_sim(pearson_sim, 'Pearson’s similarity')

# รายการของคนที่มีความชอบคล้ายกันที่สุด และรายการคนที่มีความชอบตรงข้ามกันที่สุด

cos_like_df = pd.DataFrame(cos_like)
cos_like_df.columns = ['similarity']
cos_like_df['method'] = 'cosine_sim'

cos_dislike_df = pd.DataFrame(cos_dislike)
cos_dislike_df.columns = ['similarity']
cos_dislike_df['method'] = 'cosine_sim'


pear_like_df = pd.DataFrame(pear_like)
pear_like_df.columns = ['similarity']
pear_like_df['method'] = 'pearson_sim'

pear_dislike_df = pd.DataFrame(pear_dislike)
pear_dislike_df.columns = ['similarity']
pear_dislike_df['method'] = 'pearson_sim'

sim_df = pd.concat([cos_like_df, pear_like_df])
sim_df = (sim_df.sort_values('similarity', ascending=False))

sim_df

ax = sim_df.plot(kind='bar', figsize=(40,10), title='Similarity user')
i = 0
for p in ax.patches:
  ax.annotate(sim_df.iloc[i, -1], (p.get_x() * 1.005, p.get_height() * 1.005))
  i+=1

diff_df = pd.concat([cos_dislike_df, pear_dislike_df])
diff_df = (diff_df.sort_values('similarity'))

diff_df

ax = diff_df.plot(kind='bar', figsize=(40,10), title='Opposite user')
i = 0
for p in ax.patches:
  ax.annotate(diff_df.iloc[i, -1], (p.get_x() * 1.005, p.get_height() * 1.005))
  i+=1

## แสดงรูปภาพ

corr_df = pearson_sim
pear_like = (corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))

pear_df = pd.DataFrame(pear_like)
pear_df.columns = ['similarity']
pear_df.index.names = ['user_1', 'user_2']
pear_df.reset_index(inplace=True)

# set of users
users = set(pear_df['user_1'].unique())
user_2 = set(pear_df['user_2'].unique())
users.update(user_2)
users = list(users)

from networkx import nx
from matplotlib.lines import Line2D

# Create New Graph
G = nx.Graph()

# tt = pear_df.iloc[0]
# G.add_edge(tt['user_1'], tt['user_2'], weight=tt['similarity'], color="blue")


for index, row in pear_df.iterrows():

# Similar user
  if row['similarity'] >= 0.5 :
    G.add_edge(row['user_1'], row['user_2'], weight=row['similarity'], color='b')            # blue

  elif row['similarity'] >= 0.3 :
    G.add_edge(row['user_1'], row['user_2'], weight=row['similarity'], color='g')            # green

  elif row['similarity'] >= 0.1 :
    G.add_edge(row['user_1'], row['user_2'], weight=row['similarity'], color='#CCFF99')      # lightgreen

# Opposite user
  elif row['similarity'] >= 0.05 :
    G.add_edge(row['user_1'], row['user_2'], weight=row['similarity'], color='#FFFF00')       # yellow

  elif row['similarity'] >= 0.00 :
    G.add_edge(row['user_1'], row['user_2'], weight=row['similarity'], color='#FF8000')       # orange

  elif row['similarity'] < 0 : 
    G.add_edge(row['user_1'], row['user_2'], weight=row['similarity'], color='r')             # red

G.size(100)
pos = nx.spring_layout(G)
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
weights = [G[u][v]['weight']*10 for u,v in edges]

plt.figure(3,figsize=(12,12)) 
nx.draw(G, pos, edges=edges, edge_color=colors, width=weights)
nx.draw_networkx_labels(G, pos)
plt.show()

#  แสดงรูปตาราง movie title ที่ rating สูงสุด ที่ควรแนะนำของคนที่มีความชอบคล้ายกันที่สุด ที่ควรแนะนำให้ดู
pearson_sim = random_mt.T.corr( method ='pearson' )
np.fill_diagonal(pearson_sim.values, np.nan)

# Find similar user
user_sim_df = pd.DataFrame(pearson_sim.stack())
user_sim_df.columns = ['similarity']
user_sim_df.index.names = ['user_1', 'user_2']
user_sim_df = user_sim_df.groupby(level=0).idxmax()

# Create NaN table
user_mv_rate_df = pd.DataFrame(index=user_sim_df.index, columns=['sim_userId', 'movie_title', 'movie_rating'])
user_mv_rate_df.index.name = 'userId'

# Fill similar user
for idx, row in user_sim_df.iterrows():
  user_mv_rate_df.loc[idx, 'sim_userId'] = row['similarity'][1]

# Find max user rate movie
mov_rate_df = pd.DataFrame(user_rate_mt.loc[user_mv_rate_df['sim_userId'].values].idxmax(axis=1), columns=['movie_id'])

# Find movie title
temp_movie = movie_df.copy()
temp_movie.set_index('movieId', inplace=True)
temp_movie = pd.DataFrame(temp_movie.loc[mov_rate_df['movie_id'].values, 'title'])
temp_movie.reset_index(inplace=True)
temp_movie.set_index(mov_rate_df.index, inplace=True)
mov_rate_df['movie_title'] = temp_movie['title']
mov_rate_df['movie_rating'] = np.nan

# find movie rating
sim_user_rate_mt = user_rate_mt.loc[mov_rate_df.index]
sim_user_rate_mt.drop_duplicates(inplace=True) 

temp_mov_rate_df = mov_rate_df.copy()
temp_mov_rate_df.reset_index(inplace=True)
for idx, row in temp_mov_rate_df.iterrows():
  temp_mov_rate_df.loc[idx ,'movie_rating'] = sim_user_rate_mt.loc[row['userid'], row['movie_id']]
mov_rate_df = temp_mov_rate_df.copy()

# Fill title and rating
user_mv_rate_df['movie_title'] = mov_rate_df['movie_title'].values
user_mv_rate_df['movie_rating'] = mov_rate_df['movie_rating'].values

user_mv_rate_df

# แสดงรูปตาราง movie title ที่ rating สูงสุด ของคนที่มีความชอบตรงข้ามกันที่สุด ที่ไม่ควรแนะนำให้ดู
pearson_sim = random_mt.T.corr( method ='pearson' )
np.fill_diagonal(pearson_sim.values, np.nan)

# Find opposite user
user_sim_df = pd.DataFrame(pearson_sim.stack())
user_sim_df.columns = ['opposity']
user_sim_df.index.names = ['user_1', 'user_2']
user_sim_df = user_sim_df.groupby(level=0).idxmin()

# Create NaN table
user_mv_rate_df = pd.DataFrame(index=user_sim_df.index, columns=['opp_userId', 'movie_title', 'movie_rating'])
user_mv_rate_df.index.name = 'userId'

# Fill opposite user
for idx, row in user_sim_df.iterrows():
  user_mv_rate_df.loc[idx, 'opp_userId'] = row['opposity'][1]

# Find max user rate movie
mov_rate_df = pd.DataFrame(user_rate_mt.loc[user_mv_rate_df['opp_userId'].values].idxmax(axis=1), columns=['movie_id'])

# Find movie title
temp_movie = movie_df.copy()
temp_movie.set_index('movieId', inplace=True)
temp_movie = pd.DataFrame(temp_movie.loc[mov_rate_df['movie_id'].values, 'title'])
temp_movie.reset_index(inplace=True)
temp_movie.set_index(mov_rate_df.index, inplace=True)
mov_rate_df['movie_title'] = temp_movie['title']
mov_rate_df['movie_rating'] = np.nan

# find movie rating
sim_user_rate_mt = user_rate_mt.loc[mov_rate_df.index]
sim_user_rate_mt.drop_duplicates(inplace=True) 

temp_mov_rate_df = mov_rate_df.copy()
temp_mov_rate_df.reset_index(inplace=True)
for idx, row in temp_mov_rate_df.iterrows():
  temp_mov_rate_df.loc[idx ,'movie_rating'] = sim_user_rate_mt.loc[row['userid'], row['movie_id']]
mov_rate_df = temp_mov_rate_df.copy()

# Fill title and rating
user_mv_rate_df['movie_title'] = mov_rate_df['movie_title'].values
user_mv_rate_df['movie_rating'] = mov_rate_df['movie_rating'].values

user_mv_rate_df

