#!/usr/bin/env python
# coding: utf-8

# In[1]:


import twitter
import tweepy
import pandas as pd
import numpy as np
import networkx as nx
import descartes
import geopandas as gpd
import os
from datetime import datetime
import datetime as dt
from shapely.geometry import Point, Polygon
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('float_format', '{:f}'.format)


# In[2]:


app_key = ''			#YOUR Twitter Application Key
app_key_secret = ''		#YOUR Twitter Application Secret Key


# # Tweepy

# In[3]:


auth = tweepy.AppAuthHandler(app_key, app_key_secret)
api = tweepy.API(auth)


# In[4]:


# Twitter-syntaxed queries
hashtags = ['#filmyourhospital', '#covid19', '#plandemic']			# Interesting hashtags

hashtag = hashtags[1]												# Hashtag to search

q_originals = ' -filter:replies -filter:retweets -filter:quotes'	# Not a reply, not a quote, not a retweet
q_geo = ' has:geo'													# Has some georeference field
q_english = ' lang:en'												# It is in english
q_verified = ' filter:verified'										# The user is verified

query = hashtag + q_english + q_geo


# In[5]:


def search_tweets(query, i='', date=None, save=False, users=False):
    '''
    Searches for tweets and returns them as a DataFrame.
    
    Arguments:    
        - query: String with a twitter-syntaxed query for a status search.
        - i: Any String-able type of data to add to the filenames.
        - date: A datetime object representing today and now. If None: use datetime.now()
        - save: default=False. If True: saves the searched DataFrame into .csv files under '/data'.
        - users: default=False. If True: creates a second DataFrame for the users in each tweet, saves it if save=True, and returns two DataFrames instead.
        
    Returns:
        int: The lowest id among the statuses retrieved
        
    '''
    if not date:
        date = datetime.now()
    path = 'data'
    if not os.path.exists(path):
        os.makedirs(path)

    _tweets = []
    _users = []
    for tweet in tweepy.Cursor(api.search, q=query).items(100):
        _users.append(tweet.user._json)
        _tweets.append(tweet._json)
    df_tweets = pd.DataFrame.from_dict(_tweets)
    df_users = pd.DataFrame.from_dict(_users)

    if save:
        df_tweets.to_csv(path+'/tweets_'+str(date).replace(' ', '_').replace(':', '-').split('.')[0]+'_'+str(i)+'.csv')
        if users:
            df_users.to_csv(path+'/users_'+str(date).replace(' ', '_').replace(':', '-').split('.')[0]+'_'+str(i)+'.csv')
    
    return min(df_tweets['id'])


# In[6]:


def exhaust_search(api, query, limit=None):
    '''
    Searches and saves all tweets into .csv files. The search is limited by the current remaining application requests from api.
    
    Parameters:
        - api: Tweepy.API object with an application authorization
        - query: String with a twitter-syntaxed query for a status search.
        - limit: Maximum number of API requests to make to Twitter.
        
    Returns:
        None
    '''
    
    if not limit:
        limit =  status['resources']['application']['/application/rate_limit_status']['limit']
    status = api.rate_limit_status()
    date = datetime.now()
    remaining_requests = status['resources']['application']['/application/rate_limit_status']['remaining']
    until = status['resources']['application']['/application/rate_limit_status']['reset']
    until = datetime.fromtimestamp(until)
    now = datetime.now()
    time_to_reset = until - now



    if remaining_requests <= 0:
        raise Exception(f'You have reached your API request limit. You will get another {limit} requests in:\n{time_to_reset.seconds} Seconds ({until})')

    if remaining_requests < limit:
        limit = remaining_requests
        print(f'You have {remaining_requests} requests until {until}')


    previous_max_id = ''
    for i in range(limit):
        id = search_tweets(query+previous_max_id, i+1, date, save=True)
        previous_max_id = ' max_id:' + str(id)


# In[7]:


exhaust_search(api, query)

