#!/usr/bin/env python
# coding: utf-8

# Hotel booking project is a project that will predict how likely it is for a customer to cancel their hotel booking
# 

# # data cleaning 

# In[54]:


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[55]:


#read the data

hotel = pd.read_csv("C:/Users/hp/Downloads/hotel_bookings.csv")


# In[56]:


#first 5 rows

hotel.head()


# In[57]:


# number os rows and number of columns

hotel.shape


# In[58]:


#checking for missing data

hotel.isnull().sum()


# In[59]:


#dealing with missing values

def data_clean(hotel):
    hotel.fillna(0,inplace=True)
    print(hotel.isnull().sum())


# In[60]:


#calling the function and now we dont have any null values

data_clean(hotel)


# In[61]:


#display the columns 

hotel.columns


# In[62]:


#finding the unique instances in the three categories of people

list =['adults', 'children', 'babies']
for val in list:
    print('{} has uniques values as {}'.format(val,hotel[val].unique()))


# In[63]:


#creating a filter for the zeros that exist in all the three categories

filter =(hotel["adults"]== 0) & (hotel["children"]==0) & (hotel['babies']==0)
hotel[filter]


# In[64]:


#since we lacked some columns in the previous output, now we fix that
#with set_option in pamdas

pd.set_option('display.max_column',32)


# In[65]:


filter =(hotel["adults"]== 0) & (hotel["children"]==0) & (hotel['babies']==0)
hotel[filter]


# # analysing the data

# In[66]:


#where the guests come from?
#spatial analysis

country_analysis = hotel[hotel['is_canceled']==0]['country'].value_counts().reset_index()


# In[67]:


country_analysis


# In[68]:


#rename the columns

country_analysis.columns=['country','no.of guests']


# In[69]:


country_analysis


# In[70]:


get_ipython().system('pip install folium')


# In[71]:


import folium
from folium.plugins import HeatMap


# In[72]:


basemap = folium.Map()


# In[73]:


basemap


# In[74]:


get_ipython().system('pip install plotly')


# In[75]:


import plotly.express as px


# In[76]:


map_guests = px.choropleth(country_analysis,
             locations = country_analysis['country'],
             color = country_analysis['no.of guests'],
             hover_name = country_analysis['country'],
             title='Home country of guests')
map_guests.show()


# In[77]:


#how much guests pay for hotel
#distribution of hotel type

hotel_type = hotel[hotel['is_canceled']== 0]
hotel_type.columns


# In[78]:


plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',y='adr',hue='hotel', data=hotel_type)
plt.title("price of room types per night and per person")
plt.xlabel('room type')
plt.ylabel('price(Euro)')
plt.legend()
plt.show()


# In[79]:


# how does the price per night vary over the year


# In[80]:


hotel.columns


# In[81]:


resort_hotel=hotel[(hotel['hotel']=='Resort Hotel') & (hotel['is_canceled']==0)]
city_hotel=hotel[(hotel['hotel']=='City Hotel') & (hotel['is_canceled']==0)]


# In[99]:


resort_hotel.head()


# In[100]:


#group by arrival date month and price
#to make it a df, reset index

resort_hoteldf = resort_hotel.groupby(['arrival_date_month'])['adr'].mean().reset_index()


# In[101]:


resort_hoteldf


# In[102]:


city_hoteldf = city_hotel.groupby(['arrival_date_month'])['adr'].mean().reset_index()


# In[103]:


city_hoteldf


# In[104]:


#merge the two dataframes

final = resort_hoteldf.merge(city_hoteldf, on='arrival_date_month')
final.columns = ['months','price_for_resort','price_for_city']


# In[105]:


final


# In[106]:


#sort the months using 

get_ipython().system('pip install sorted-months-weekdays')


# In[107]:


get_ipython().system('pip install sort-dataframeby-monthorweek')


# In[108]:


import sort_dataframeby_monthorweek as sd


# In[109]:


def sort_data(df,colname):
    return sd.Sort_Dataframeby_Month(df,colname)
    


# In[110]:


final = sort_data(final,'months')
final


# In[111]:


#visuals.
#line plot

px.line(final,x='months',y=['price_for_resort', 'price_for_city'],
       title='room price overnight per month')


# In[112]:


final.columns


# In[115]:


#analysis demands of hotels


# In[116]:


rush_resort = resort_hotel['arrival_date_month'].value_counts().reset_index()
rush_resort.columns = ['months','no.of guests']
rush_resort


# In[117]:


rush_city = city_hotel['arrival_date_month'].value_counts().reset_index()
rush_city.columns = ['months','no.of guests']
rush_city


# In[118]:


#merge dataframes

final_rush = rush_resort.merge(rush_city,on='months')
final_rush.columns = ['months','no of guests in resort','no of guest in city']
final_rush


# In[119]:


#hierachy of my months
final_rush = sort_data(final_rush,'months')
final_rush


# In[120]:


#we need trend, so we go for line plot

px.line(final_rush,x='months',y= ['no of guests in resort', 'no of guest in city'],
       title='total no of guest per months')


# # machine learning
# 

# In[121]:


hotel.head()


# In[122]:


#find correlation
hotel.corr()


# In[123]:


#correlation with respet to is cancelled

co_relate = hotel.corr()['is_canceled']
co_relate


# In[124]:


#finding the most important features

co_relate.abs().sort_values(ascending=False)


# In[125]:


#
hotel.groupby('is_canceled')['reservation_status'].value_counts()


# In[126]:


#exclude unnecessary features

list_not = ['days_in_waiting_list ','arrival_date_year ']


# In[127]:


#fetch numerical features we have
#using a list comprehension

num_features = [col for col in hotel.columns if hotel[col].dtype != 'object' and col not in list_not]
num_features


# In[128]:


hotel.columns


# In[129]:


cat_not = ['arrival_date_year','country','assigned_room_type','booking_changes', 'reservation_status','days_in_waiting_list']


# In[130]:


cat_features = [col for col in hotel.columns if hotel[col].dtype == 'object' and col not in cat_not]


# In[131]:


cat_features


# In[132]:


#extracting derived features 




# In[133]:


cat_data = hotel[cat_features]


# In[134]:


cat_data.dtypes


# In[135]:


#when you want to block the warning
import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[136]:


cat_data['reservation_status_date'] = pd.to_datetime(cat_data['reservation_status_date'])


# In[137]:


#creating different columns for month day and year

cat_data['year'] = cat_data['reservation_status_date'].dt.year
cat_data['month'] = cat_data['reservation_status_date'].dt.month
cat_data['day'] = cat_data['reservation_status_date'].dt.day


# In[138]:


#drop the column with the combination of the data

cat_data.drop('reservation_status_date', axis =1, inplace=True)


# In[139]:


cat_data.dtypes


# In[140]:


cat_data['cancellation']=hotel['is_canceled']


# In[ ]:





# In[141]:


cat_data.dtypes


# In[142]:


#applying feature encoding
cat_data.head()


# In[143]:


##mean encoding technique

cat_data['market_segment'].unique()


# In[144]:


col_enc = cat_data.columns[0:8]


# In[145]:


for col in col_enc:
    print(cat_data.groupby([col])['cancellation'].mean().to_dict())
    print('\n')


# In[146]:


for col in col_enc:
    dict = cat_data.groupby([col])['cancellation'].mean().to_dict()
    cat_data[col] = cat_data[col].map(dict)


# In[147]:


cat_data.head()


# In[148]:


num_features


# In[149]:


entire_df = pd.concat([cat_data,hotel[num_features]], axis=1)


# In[150]:


entire_df.head()


# In[151]:


#
entire_df.drop('cancellation', axis=1, inplace=True)


# In[152]:


entire_df.shape


# In[153]:


#handling outliers

entire_df.head()


# In[154]:


#distribution of lead time

sns.distplot(entire_df['lead_time'])


# In[155]:


#find the log of these 

def handle_outlier(col):
    entire_df[col] = np.log1p(entire_df[col])


# In[156]:


handle_outlier('lead_time')


# In[157]:


sns.distplot(entire_df['lead_time'])


# In[158]:


sns.distplot(entire_df['adr'])


# In[159]:


handle_outlier('adr')


# In[160]:


sns.distplot(entire_df['adr'].dropna())


# In[161]:


sns.distplot(entire_df['adr'])


# In[162]:


#applying feature importance
#most important features

entire_df.isnull().sum()


# In[163]:


entire_df.dropna(inplace=True)


# In[164]:


#dependent and independent feature

#dependent feature
y = entire_df['is_canceled']

#independent features
x=entire_df.drop('is_canceled',axis=1)


# In[165]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[166]:


feature_selmodel = SelectFromModel(Lasso(alpha=0.005, random_state=0))


# In[167]:


feature_selmodel.fit(x,y)


# In[168]:



feature_selmodel.get_support()


# In[169]:


cols = x.columns


# In[170]:


selected_feat = cols[feature_selmodel.get_support()]


# In[171]:


print('total features {}'. format(x.shape[1]))
print('selected features {}'.format (len(selected_feat)))


# In[172]:


x = x[selected_feat]


# # logistic regression

# In[173]:


#applying machine learning
#cross validation of data


# In[174]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[188]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)


# In[189]:


logreg = LogisticRegression()


# In[190]:


logreg.fit(X_train,y_train)


# In[191]:


y_pred = logreg.predict(X_test)


# In[192]:


y_pred


# In[184]:


from sklearn.metrics import confusion_matrix


# In[185]:


confusion_matrix(y_test, y_pred)


# In[193]:


from sklearn.metrics import accuracy_score


# In[194]:


accuracy_score(y_test,y_pred)


# In[195]:


from sklearn.model_selection import cross_val_score


# In[196]:


score = cross_val_score(logreg, x, y, cv=10)


# In[197]:


score.mean()


# # applying various algorithmn on this data.

# In[199]:


#importing the models from sklearn

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[200]:


#initializing the model

models = []

models.append(('LogisticRegression', LogisticRegression()))
models.append(('Naive bayes', GaussianNB()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('Decision tree ',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))


# In[201]:


#fit the models

for name,model in models:
    print(name)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(predictions, y_test))
    print('\n')
    
    print(accuracy_score(predictions, y_test))
    print('\n')


# In[ ]:




