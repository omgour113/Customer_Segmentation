#!/usr/bin/env python
# coding: utf-8

# In[71]:


# handle table-like data and matrices
import pandas as pd
import numpy as np

# visualisation
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().system('pip install missingno')
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# preprocessing
from sklearn.preprocessing import StandardScaler

# pca
from sklearn.decomposition import PCA

# clustering
get_ipython().system('pip install yellowbrick')
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, AgglomerativeClustering

# evaluations
from sklearn.metrics import confusion_matrix

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# to display the total number columns present in the dataset
pd.set_option('display.max_columns', None)


# In[2]:


data = pd.read_csv('marketing_campaign.csv', sep="\t")


# In[3]:


data.isnull().sum()


# In[4]:


msno.matrix(data);


# In[5]:


data = data.dropna()


# In[6]:


data.isnull().sum()


# In[7]:


data.duplicated().sum()


# In[8]:


data.info()


# In[9]:


data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])


# In[10]:


print("The newest customer's enrolment date in the records:", max(data['Dt_Customer']))
print("The oldest customer's enrolment date in the records:", min(data['Dt_Customer']))


# In[11]:


data['Age'] = 2015 - data['Year_Birth']


# In[12]:


data['Spent'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']


# In[13]:


data['Living_With'] = data['Marital_Status'].replace({'Married':'Partner', 'Together':'Partner', 'Absurd':'Alone', 'Widow':'Alone', 'YOLO':'Alone', 'Divorced':'Alone', 'Single':'Alone'})


# In[14]:


data['Children'] = data['Kidhome'] + data['Teenhome']


# In[15]:


data['Family_Size'] = data['Living_With'].replace({'Alone': 1, 'Partner':2}) + data['Children']


# In[16]:


data['Is_Parent'] = np.where(data.Children > 0, 1, 0)


# In[17]:


data['Education'] = data['Education'].replace({'Basic':'Undergraduate', '2n Cycle':'Undergraduate', 'Graduation':'Graduate', 'Master':'Postgraduate', 'PhD':'Postgraduate'})


# In[18]:


to_drop = ['Marital_Status', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Year_Birth', 'ID']
data = data.drop(to_drop, axis=1)


# In[19]:


data.head(3)


# In[20]:


data.shape


# In[21]:


data.info()


# In[22]:


data.describe()


# data.describe(include=object).T

# sns.pairplot(data , vars=['Spent','Income','Age'] , hue='Children', palette='husl');

# In[23]:


plt.figure(figsize=(13,8))
sns.scatterplot(x=data[data['Income']<600000]['Spent'], y=data[data['Income']<600000]['Income'], color='#cc0000');


# In[24]:


plt.figure(figsize=(13,8))
sns.scatterplot(x=data['Spent'], y=data['Age']);


# In[25]:


plt.figure(figsize=(13,8))
sns.histplot(x=data['Spent'], hue=data['Education']);


# In[26]:


data['Education'].value_counts().plot.pie(explode=[0.1,0,0], autopct='%1.1f%%', shadow=True, figsize=(8,8), colors=sns.color_palette('bright'));


# In[27]:


plt.figure(figsize=(13,8))
sns.distplot(data.Age, color='violet');


# In[28]:


plt.figure(figsize=(13,8))
sns.distplot(data.Income, color='Yellow');


# In[29]:


plt.figure(figsize=(13,8))
sns.distplot(data.Spent, color='#ff9966');


# In[30]:


fig = make_subplots(rows=1, cols=3)

fig.add_trace(go.Box(y=data['Age'], notched=True, name='Age', marker_color = '#6699ff', 
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 2)

fig.add_trace(go.Box(y=data['Income'], notched=True, name='Income', marker_color = '#ff0066', 
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 1)

fig.add_trace(go.Box(y=data['Spent'], notched=True, name='Spent', marker_color = 'lightseagreen', 
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 3)

fig.update_layout(title_text='<b>Box Plots for Numerical Variables<b>')

fig.show()


# In[31]:


data.head(1)


# In[32]:


numerical = ['Income', 'Recency', 'Age', 'Spent']


# In[33]:


def detect_outliers(d):
  for i in d:
    Q3, Q1 = np.percentile(data[i], [75 ,25])
    IQR = Q3 - Q1

    ul = Q3+1.5*IQR
    ll = Q1-1.5*IQR

    outliers = data[i][(data[i] > ul) | (data[i] < ll)]
    print(f'*** {i} outlier points***', '\n', outliers, '\n')


# In[34]:


detect_outliers(numerical)


# In[35]:


ata = data[(data['Age']<100)]
data = data[(data['Income']<600000)]


# In[36]:


data.shape


# In[37]:


categorical = [var for var in data.columns if data[var].dtype=='O']


# In[38]:


# check the number of different labels
for var in categorical:
    print(data[var].value_counts() / np.float(len(data)))
    print()
    print()


# In[39]:


categorical


# In[40]:


data['Living_With'].unique()


# In[41]:


data['Education'] = data['Education'].map({'Undergraduate':0,'Graduate':1, 'Postgraduate':2})


# In[42]:


data['Living_With'] = data['Living_With'].map({'Alone':0,'Partner':1})


# In[43]:


data.dtypes


# In[44]:


data.head(3)


# In[45]:


corrmat = data.corr()

plt.figure(figsize=(20,20))  
sns.heatmap(corrmat, annot = True, cmap = 'mako', center = 0)


# In[46]:


data_old = data.copy()


# In[47]:


# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
data = data.drop(cols_del, axis=1)


# In[48]:


scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)


# In[49]:


data.head(3)


# In[50]:


p = PCA(n_components=3)
p.fit(data)


# In[51]:


W = p.components_.T
W


# In[52]:


pd.DataFrame(W, index=data.columns, columns=['W1','W2','W3'])


# In[53]:


p.explained_variance_


# In[54]:


p.explained_variance_ratio_


# In[55]:


pd.DataFrame(p.explained_variance_ratio_, index=range(1,4), columns=['Explained Variability'])


# In[56]:


p.explained_variance_ratio_.cumsum()


# In[57]:


sns.barplot(x = list(range(1,4)), y = p.explained_variance_, palette = 'GnBu_r')
plt.xlabel('i')
plt.ylabel('Lambda i');


# In[58]:


data_PCA = pd.DataFrame(p.transform(data), columns=(['col1', 'col2', 'col3']))


# In[59]:


data_PCA.describe().T


# In[60]:


x = data_PCA['col1']
y = data_PCA['col2']
z = data_PCA['col3']

fig = plt.figure(figsize=(13,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, c='darkred', marker='o')
ax.set_title('A 3D Projection of Data In the Reduced Dimension')
plt.show()


# In[61]:


Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(data_PCA)
Elbow_M.show();


# In[62]:


AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(data_PCA)
data_PCA['Clusters'] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
data['Clusters'] = yhat_AC
data_old['Clusters'] = yhat_AC


# In[63]:


fig = plt.figure(figsize=(13,8))
ax = plt.subplot(111, projection='3d', label='bla')
ax.scatter(x, y, z, s=40, c=data_PCA['Clusters'], marker='o', cmap='Set1_r')
ax.set_title('Clusters')
plt.show()


# In[64]:


pal = ['gold','#cc0000', '#ace600','#33cccc']
plt.figure(figsize=(13,8))
pl = sns.countplot(x=data['Clusters'], palette= pal)
pl.set_title('Distribution Of The Clusters')
plt.show()


# In[65]:


plt.figure(figsize=(13,8))
pl = sns.scatterplot(data=data_old, x=data_old['Spent'], y=data_old['Income'], hue=data_old['Clusters'], palette= pal)
pl.set_title("Cluster's Profile Based on Income and Spending")
plt.legend();


# In[66]:


plt.figure(figsize=(13,8))
pl = sns.swarmplot(x=data_old['Clusters'], y=data_old['Spent'], color="#CBEDDD", alpha=0.7)
pl = sns.boxenplot(x=data_old['Clusters'], y=data_old['Spent'], palette=pal)
plt.show();


# In[67]:


data_old['Total_Promos'] = data_old['AcceptedCmp1']+ data_old['AcceptedCmp2']+ data_old['AcceptedCmp3']+ data_old['AcceptedCmp4']+ data_old['AcceptedCmp5']

plt.figure(figsize=(13,8))
pl = sns.countplot(x=data_old['Total_Promos'], hue=data_old['Clusters'], palette= pal)
pl.set_title('Count Of Promotion Accepted')
pl.set_xlabel('Number Of Total Accepted Promotions')
plt.legend(loc='upper right')
plt.show();


# In[68]:


plt.figure(figsize=(13,8))
pl=sns.boxenplot(y=data_old['NumDealsPurchases'],x=data_old['Clusters'], palette= pal)
pl.set_title('Number of Deals Purchased');


# In[70]:


Personal = ['Kidhome', 'Teenhome', 'Age', 'Children', 'Family_Size', 'Is_Parent', 'Education', 'Living_With']

for i in Personal:
    plt.figure(figsize=(13,8))
    sns.jointplot(x=data_old[i], y=data_old['Spent'], hue=data_old['Clusters'], kind='kde', palette=pal);

