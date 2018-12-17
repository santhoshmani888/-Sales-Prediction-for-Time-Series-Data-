
# coding: utf-8

# # Import libraries

# In[91]:

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime

get_ipython().magic('matplotlib inline')
sns.set()


# In[92]:

import time
from  sklearn.metrics import mean_squared_error
start_time = time.time()
from math import *
import lightgbm as lgb



# # Read dataset from work directory

# In[93]:

sales_train = pd.read_csv('sales_train.csv.gz')
sales_test = pd.read_csv('test.csv.gz')
shops = pd.read_csv('shops.csv')
items = pd.read_csv('items.csv')
item_cats = pd.read_csv('item_categories.csv')
sampleSubmission = pd.read_csv('sample_submission.csv.gz')

sales_train.name = 'sales_train'
sales_test.name = 'sales_test'
shops.name = 'shops'
items.name = 'items'
item_cats.name = 'item_cats'


# In[94]:

dflist = [sales_train, sales_test, shops, items, item_cats]
#show_info(dflist)
print('training set: ', sales_train.shape)
print('test set: ', sales_test.shape)
print('num of shops: ', shops.shape)
print('num of items: ',items.shape)
print('num of item categories: ',item_cats.shape)


# # building dummy features for test dataset-34th month

# In[95]:

sales_test['ID'] = sales_test.index
sales_train['ID'] = sales_train.index
sales_test['set'] = 'test'
sales_train['set'] = 'train'
sales_test['date_block_num'] = 34
sales_test['item_price'] = np.nan
sales_test['item_cnt_day'] = np.nan
sales_test['date'] = '30.11.2015'


# In[96]:

sales_test.head(3)


# In[97]:

sales_train = sales_train[['ID', 'set', 'date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']]
sales_test = sales_test[['ID', 'set', 'date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']]


# In[98]:

sales_test.head(3)


# In[99]:

data = sales_train.append(sales_test)


# # merge shops,items and item_cat

# In[100]:

data = data.merge(shops, how = 'left', on = 'shop_id')
data = data.merge(items, how = 'left', on = 'item_id')
data = data.merge(item_cats, how = 'left', on = 'item_category_id')


# In[101]:

data.head(3)


# # extract time features

# In[102]:

data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day'] = data['date'].dt.day
data['weekday'] = data['date'].dt.weekday


# In[103]:

data.head()


# In[104]:

data.to_csv('newsampple', header=True, index=False, encoding='utf-8')


# # highly skewed target data

# In[105]:

cnt_Data = data[(data.set == 'train')]['item_cnt_day']
sns.distplot(cnt_Data)


# In[106]:

data['item_cnt_day_dup'] = data['item_cnt_day'] # keep for record
data['item_cnt_day'] = data['item_cnt_day_dup'].clip(0, 20)#clip target 20 id


# In[107]:

data['item_price'].describe()


# # Removing data outlier

# In[108]:

data[data['item_price'] == -1]


# In[109]:

meanPrice = data[data.item_id == 2973]['item_price'].mean()
data.loc[data.index == 484683, 'item_price'] = meanPrice


# In[110]:

data[data.item_price > np.nanpercentile(data.item_price, 99.9)].groupby(['item_id'])['item_price'].agg(["mean", "max", "min", "std", "count"])


# In[111]:

data[data.item_id == 6066]


# In[112]:

data = data.drop([1163158], axis= 0)


# # create features

# In[113]:

data['revenue'] = data['item_cnt_day'] * data['item_price']


# In[114]:

data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')


# In[115]:

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month


# In[116]:

data.head(3)


# In[117]:

data.to_csv('newsam.csv', header=True, index=False, encoding='utf-8')


# # Month Vs Revenue comparison

# In[118]:

plt.rcParams['figure.figsize'] = (15, 12)


# In[119]:

monthlyRev = pd.DataFrame(data.groupby(["month", "year"], as_index=False)["revenue"].sum())
monthlyRev.head()


g = sns.FacetGrid(data = monthlyRev.sort_values(by="month"), hue = "year", size = 5, legend_out=True)
g = g.map(plt.plot, "month", "revenue")
g.add_legend()


# In[120]:

# change data type 
def cast_dtype(df, cols, dtype):
    df[cols] = df[cols].astype(dtype)
    
    return df


# In[121]:

def downcast_dtypes(df):

    # Select columns to downcast to reduce memory
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast to 32 bits
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df


# # Month Vs Revenue comparison

# In[122]:

vizdata = data.groupby(['year','month'])['revenue'].sum()
vizdata = pd.DataFrame(vizdata)
vizdata.reset_index(inplace=True)
cast_dtype(vizdata, ['year','month'], object);


# In[123]:

sns.barplot(x='month', y='revenue', hue='year', data=vizdata)


# # Week vs revenue

# In[124]:

vizdata = data.groupby(['weekday'])['revenue'].sum()
vizdata = pd.DataFrame(vizdata)
vizdata.reset_index(inplace=True)


# In[125]:

plt.rcParams['figure.figsize']=(10,10)


# In[126]:

sns.barplot(x='weekday', y='revenue', data=vizdata)


# # item category vs revenue

# In[127]:

vizdata = data.groupby(['item_category_id'])['revenue'].sum()
vizdata = pd.DataFrame(vizdata)
vizdata.reset_index(inplace=True)
vizdata.nlargest(5, 'revenue').merge(item_cats, how = 'left', on = 'item_category_id')


# In[128]:

plot = sns.barplot(x='item_category_id', y='revenue', data=vizdata)


# # plot shop vs revenue

# In[129]:

vizdata = data.groupby(['shop_id'])['revenue'].sum()
vizdata = pd.DataFrame(vizdata)
vizdata.reset_index(inplace=True)
vizdata.nlargest(5, 'revenue').merge(shops, how = 'left', on = 'shop_id')


# In[130]:

plot = sns.barplot(x='shop_id', y='revenue', data=vizdata)


# In[131]:

data = downcast_dtypes(data)


# In[132]:

data.info()


# In[133]:

index_cols = ['shop_id', 'item_id', 'date_block_num']


# In[134]:

from itertools import product #This tool computes the cartesian product of input iterables
casc = [] 
for block_num in data['date_block_num'].unique():
    cur_shops = data.loc[data['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = data.loc[data['date_block_num'] == block_num, 'item_id'].unique()
    casc.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))


# In[135]:

casc = pd.DataFrame(np.vstack(casc), columns = index_cols, dtype=np.int32)


# In[136]:

casc.head(10)


# In[137]:

summary = data.groupby(index_cols, as_index=False).agg({'item_cnt_day': ['sum']})
summary.columns = ['shop_id', 'item_id', 'date_block_num', 'target_item_cnt']



# In[138]:

summary.head(10)


# In[139]:

all_data = pd.merge(casc, summary, how='left', on=index_cols).fillna(0)


# In[140]:

all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
all_data['target_item_cnt'] = all_data['target_item_cnt'].clip(0, 20, axis=0)


# In[141]:

all_data = all_data.merge(shops, how = 'left', on = 'shop_id')
all_data = all_data.merge(items, how = 'left', on = 'item_id')
all_data = all_data.merge(item_cats, how = 'left', on = 'item_category_id')


# In[142]:

all_data.head()


# In[143]:

index_cols = ['shop_id', 'date_block_num', 'item_category_id']
summary = data.groupby(index_cols, as_index=False).agg({'revenue': ['sum']})
summary.columns = ['shop_id', 'date_block_num', 'item_category_id', 'category_revenue']


# In[144]:

all_data = pd.merge(all_data, summary, how='left', on= ['shop_id', 'date_block_num', 'item_category_id']).fillna(0)


# In[145]:

all_data.head()


# In[146]:

all_data = all_data.drop(['shop_name','item_name'],axis=1)


# In[147]:

index_cols = ['shop_id', 'date_block_num']
summary = data.groupby(index_cols, as_index=False).agg({'item_cnt_day': ['sum']})
summary.columns = ['shop_id', 'date_block_num', 'shop_item_cnt']


# In[148]:

all_data = pd.merge(all_data, summary, how='left', on= ['shop_id', 'date_block_num'])


# In[149]:

index_cols = ['shop_id', 'item_id', 'date_block_num']
categorical_cols = ['yymm', 'month', 'year', 'set', 'shop_name',                    'item_name', 'item_category_id', 'item_category_name']
numeric_cols = list(all_data.columns.difference(index_cols + categorical_cols))


# In[150]:

numeric_cols


# # Creating Lag features

# In[151]:

shift_range = [1]#list containing different shifts

for month_shift in shift_range:
    train_shift = all_data[index_cols + numeric_cols].copy()
    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    
    newNames = lambda x: '{}_lag_{}'.format(x, month_shift) if x in numeric_cols else x # Rename columns to target_lag_1  target_item_lag_1 target_shop_lag_1
    train_shift = train_shift.rename(columns = newNames)

    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0) 


# In[152]:

all_data=all_data.drop(['item_category_name'],axis=1)


# In[153]:

summary = data[['date_block_num', 'month', 'year', 'set']].drop_duplicates().reset_index(drop = True)
summary.head()


# In[154]:

all_data = pd.merge(all_data, summary, how='left', on= ['date_block_num'])


# In[155]:

all_data.head(3)


# # correalation of features in final dataframe

# In[156]:

fig = plt.figure(figsize=(18, 14))
corr = all_data.corr()
c = plt.pcolor(corr)
plt.yticks(np.arange(0.5, len(corr.index), 1), corr.index)
plt.xticks(np.arange(0.5, len(corr.columns), 1), corr.columns)
fig.colorbar(c)



# In[157]:

to_drop_cols = list(all_data.select_dtypes(include = ['object']).columns) + ['target_item_cnt']


# In[158]:

to_drop_cols


# In[159]:

last_block = 34
X_train_full = all_data.loc[(all_data['date_block_num'] < last_block)].drop(to_drop_cols, axis=1)


# In[160]:

y_train_full = all_data.loc[(all_data['date_block_num'] < last_block), 'target_item_cnt'].values


# In[161]:

X_test_full = all_data.loc[(all_data['date_block_num'] == last_block)]
X_test_full = pd.merge(X_test_full, sales_test, how='left', on=['shop_id','item_id'])
X_test_full = X_test_full.sort_values(['ID'])
X_test_full = X_test_full.reset_index(drop = True)
ID = X_test_full['ID']
X_test_full = X_test_full.drop(['target_item_cnt']+['ID'], axis = 1)


# In[162]:

import random
last_block = 33
sampleSize = 0.2
random.seed(a = 123)
sample= list(all_data.item_id.unique())
sample= list(np.random.choice(sample, size= int(len(sample)*sampleSize), replace=False, p=None))


# In[163]:

X_train_1_index = X_train_full.loc[(X_train_full['date_block_num'] <  last_block) & (X_train_full['item_id'].isin(sample))].index
X_val_1_index = X_train_full.loc[(X_train_full['date_block_num'] == last_block) & (X_train_full['item_id'].isin(sample))].index


# In[173]:

X_train=X_train_full.iloc[X_train_1_index]
X_val=X_train_full.iloc[X_val_1_index ]


# In[174]:

y_train = y_train_full[X_train_1_index]
y_val =  y_train_full[X_val_1_index]


# In[175]:

X_train.shape,y_train.shape


# In[187]:

preds = []

lgb_params = {
              'feature_fraction': 0.9,
              'metric': 'rmse',
              'nthread':1,
              'min_data_in_leaf': 2**7,
              'bagging_fraction': 0.7,
              'learning_rate': 0.09,
              'objective': 'mse',
              'bagging_seed': 2**7,
              'num_leaves': 2**7,
              'bagging_freq':1,
              'verbose':0
              }

print('Training Model %d: %s'%(len(preds), 'lightgbm'))
start = time.perf_counter()
estimator = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 300)
pred_test = estimator.predict(X_val)
preds.append(pred_test)
pred_train = estimator.predict(X_train)
print('Train RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(y_train, pred_train))))
print('validation  RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(y_val, pred_test))))
run = time.perf_counter() - start
print('{} runs for {:.2f} seconds.'.format('lightgbm', run))
print()


# In[177]:

lgb.plot_importance(estimator)


# In[178]:

#plot of actual and prediction value


# In[179]:

plt.plot(y_val)
plt.plot(pred_test, color='red')
plt.rcParams['figure.figsize'] = [15, 30]
plt.show()


# In[180]:

def submission(model, X_test):
   
    # model prediction
    pred = model.predict(X_test)
   

    # create prediction dataframe
    
    output = pd.DataFrame() 
    output['ID'] = ID
    output['item_cnt_month'] = pred
    print(output.head())


    print('result.csv')
    
    output.to_csv(header=True, index=False, path_or_buf = 'submission.csv')
    
    return None


# In[181]:

X_test_full.head()


# In[182]:

X_test_full=X_test_full[['shop_id', 'item_id', 'date_block_num_x', 'item_category_id','category_revenue', 'shop_item_cnt', 'category_revenue_lag_1','shop_item_cnt_lag_1', 'target_item_cnt_lag_1', 'month', 'year']]


# In[183]:

submission(estimator, X_test_full)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



