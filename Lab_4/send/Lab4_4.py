#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ตอนที่ 4: การจัดกลุ่ม User ที่มีความชอบคล้ายกันด้วยเทคนิค k-mean และ Gaussian Mixture
### สร้างข้อมูลความชอบของ user ตาม genre ของ movie ในรูปของ genre_rating_matrix
m_rate_user = user_rate_mt[user_rate_mt.gt(0)]
user_rate_mov_list = list()

for i in m_rate_user.index:
  df = pd.DataFrame(m_rate_user.loc[i]).T
  df.dropna(axis=1, inplace=True)
  user_rate_mov_list.append(df)

genre_rate_df_list = list()
genre_rate_df = genre_df.copy()
genre_rate_df.reset_index(drop=True, inplace=True)
genre_rate_df.set_index('movieId', inplace=True)
genre_rate_df = genre_rate_df.astype(float)

for i in range(len(user_rate_mov_list)) :
  df = genre_rate_df.copy()
  df = df.loc[user_rate_mov_list[i].columns]
  genre_rate_df_list.append(df)

for i, df in enumerate(genre_rate_df_list):
  for col in user_rate_mov_list[i].columns:
    genre_rate_df_list[i].loc[col].replace({
        1 : user_rate_mov_list[i][col].values[0],
        0 : np.nan}, inplace=True)

# find mean
mean_user_genre_list = list()
for i in range(len(genre_rate_df_list)):
  df = pd.DataFrame({ i : genre_rate_df_list[i].mean() }).T
  df.fillna(0, inplace=True)
  mean_user_genre_list.append(df)

genre_rating_mt = pd.concat(mean_user_genre_list)
genre_rating_mt.index.name = 'userId'
genre_rating_mt.columns.name = 'genres'
genre_rating_mt

###  สำหรับ K-mean model ให้กำหนดจำนวน n_cluster

t_df, test_df = train_test_split(genre_rating_mt, test_size=0.15, random_state=42)
train_df, valid_df = train_test_split(t_df, test_size=0.10, random_state=42)

print(train_df.shape)
print(test_df.shape)
print(valid_df.shape)

kmeans = KMeans(n_clusters=3, random_state=0).fit(train_df)
pred_y_train = kmeans.labels_
pred_y_train

pred = kmeans.predict(valid_df)
pred

kmean_center = kmeans.cluster_centers_
kmean_center

# แสดงรูปภาพผลลัพธ์ของการจัดกลุ่ม กำหนดให้เทียบรูปภาพอย่างน้อย n_cluster อย่างน้อย 3 ค่า
col_1 = 9
col_2 = 13



f = plt.figure(figsize=(24,8))
ax = f.add_subplot(131)
n_clusters = 3
kmeans = KMeans(n_clusters)
kmeans.fit(train_df)
y_kmeans = kmeans.predict(valid_df)
centers = kmeans.cluster_centers_
ax.scatter(valid_df.iloc[:, col_1], valid_df.iloc[:, col_2], c=y_kmeans, s=50, cmap='viridis')
ax.scatter(centers[:, col_1], centers[:, col_2], c=range(n_clusters),cmap='viridis', s=400, alpha=0.5);
ax.title.set_text('number of clusters = 3')

ax2 = f.add_subplot(132)
n_clusters = 4
kmeans = KMeans(n_clusters)
kmeans.fit(train_df)
y_kmeans = kmeans.predict(valid_df)
centers = kmeans.cluster_centers_
ax2.scatter(valid_df.iloc[:, col_1], valid_df.iloc[:, col_2], c=y_kmeans, s=50, cmap='viridis')
ax2.scatter(centers[:, col_1], centers[:, col_2],  c=range(n_clusters),cmap='viridis', s=400, alpha=0.5);
ax2.title.set_text('number of clusters = 4')

ax3 = f.add_subplot(133)
n_clusters = 5
kmeans = KMeans(n_clusters)
kmeans.fit(train_df)
y_kmeans = kmeans.predict(valid_df)
centers = kmeans.cluster_centers_
ax3.scatter(valid_df.iloc[:, col_1], valid_df.iloc[:, col_2], c=y_kmeans, s=50, cmap='viridis')
ax3.scatter(centers[:, col_1], centers[:, col_2],  c=range(n_clusters),cmap='viridis', s=400, alpha=0.5);
ax3.title.set_text('number of clusters = 5')
###  สำหรับ Gaussian Mixture Model

import matplotlib as mpl
from scipy import linalg
from sklearn.mixture import GaussianMixture 
from matplotlib.patches import Ellipse
from pylab import *

gm = GaussianMixture(n_components=3, random_state=0).fit(train_df)

gm_wei = np.round(gm.weights_, 2)
gm_wei

gm_mean = np.round(gm.means_, 2)
gm_mean

gm_cov = np.round(gm.covariances_, 2)
gm_cov

pred = gm.predict(valid_df)
pred

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, color='r', **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, 
                             angle,color=color, **kwargs))
        
def plot_gmm(gmm, X, col_id0, col_id1, n_components, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.predict(X)
    if label:
        ax.scatter(X.iloc[:, col_id0], X.iloc[:, col_id1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X.iloc[:, col_id0], X.iloc[:, col_id1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    
    cmap = cm.get_cmap('viridis', n_components)    
    color_list = []
    for i in range(cmap.N):
      rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
      color_list.append(matplotlib.colors.rgb2hex(rgb))
    for pos, covar, w, c  in zip(gmm.means_, gmm.covariances_, gmm.weights_,color_list):
        draw_ellipse(np.array([pos[col_1],pos[col_2]]), np.array([[covar[col_1][col_1],covar[col_1][col_2]],[covar[col_2][col_1],covar[col_2][col_2]]]), alpha=w * w_factor, color=c)

gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full').fit(train_df)
plot_gmm(gmm, valid_df, col_1, col_2, 3)

col_1 = 9
col_2 = 14


n_components = 3
f = plt.figure(figsize=(24,8))
ax = f.add_subplot(131)
gmm = GaussianMixture(n_components, random_state=42, covariance_type='full').fit(train_df)
plot_gmm(gmm, valid_df, col_1, col_2,n_components)
ax.title.set_text('number of clusters = 3')

ax2 = f.add_subplot(132)
n_components=4
gmm = GaussianMixture(n_components, random_state=42, covariance_type='full').fit(train_df)
plot_gmm(gmm, valid_df, col_1, col_2, n_components)
ax2.title.set_text('number of clusters = 4')

ax3 = f.add_subplot(133)
n_components = 5
gmm = GaussianMixture(n_components, random_state=42, covariance_type='full').fit(train_df)
plot_gmm(gmm, valid_df, col_1, col_2, n_components)
ax3.title.set_text('number of clusters = 5')

