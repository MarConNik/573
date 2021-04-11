#!/usr/bin/env python
# coding: utf-8

# In[1]:


## This script compares Spanglish_test_conll_unlabeled 
## with tweets found in dev, train, and sp_train
## to see if the labeled data exists


# In[2]:


train_con_fi = '../data/Semeval_2020_task9_data/Spanglish/train.conll'
sp_train_con_fi = '../data/Semeval_2020_task9_data/Spanglish/Spanglish_train.conll'
sp_test_fi = '../data/Semeval_2020_task9_data/Spanglish/Spanglish_test_conll_unlabeled.txt'
sp_dev_fi = '../data/Semeval_2020_task9_data/Spanglish/Spanglish_dev.conll'


# In[3]:


def structure_data(fi):
    print('Now on {}'.format(fi.split('//')[-1]))
    with open(fi, 'r', encoding='utf-8') as f:
        lines = [x.replace('\n', '') for x in f.readlines()]
        
    d = {}
    cur_tweet = []
    cur_num = '1'  # all files start with 1
    cur_sentiment = 'positive'   # all files start positive
    for i in range(1, len(lines)):
        l = lines[i]
        data = l.split('\t')
        if l == '' or l == '\t':
            
            tweet_tokes = [x[0].lower() for x in cur_tweet]
            tweet_langs = [x[1] for x in cur_tweet]
            
            # unlabeled doesn't have sentiments
            if 'unlabeled' in fi:
                d[cur_num] = {'tweet_tokens':tweet_tokes, 'tweet_langs':tweet_langs}
            else:
                d[cur_num] = {'tweet_tokens':tweet_tokes, 'tweet_langs':tweet_langs,'sentiment':cur_sentiment}
        elif data[0].startswith('# sent_enum'):
            cur_num = data[0].split('=')[-1].strip()
            cur_sentiment = data[1]
            cur_tweet = []
        
        # other format
        elif data[0] == 'meta':
            if 'unlabeled' in fi:
                cur_num = data[1]
                cur_tweet = []
            else:
                cur_num = data[0].split('=')[-1].strip()
                cur_sentiment = data[1]
                cur_tweet = []
        
        else:
            cur_tweet.append(data)
    return d


# In[4]:


train_con_d = structure_data(train_con_fi)
sp_train_con_d = structure_data(sp_train_con_fi)
sp_test_d = structure_data(sp_test_fi)
sp_dev_d = structure_data(sp_dev_fi)


# ## Compare train_conll to sp_test

# In[5]:


# uses heuristics: 
# if two tweets have identical first 5 words, they are assumed to be same tweet
# if two tweets have identical last 3 words and the same length exactly, also assumed identical
# this helps with the shitty encoding


# In[6]:


com = {}
count = 0
for k in sp_test_d:
    test_tokes = sp_test_d[k]['tweet_tokens']
    found = False
    for k2 in sp_train_con_d:
        sp_train_con_tokes = sp_train_con_d[k2]['tweet_tokens']
        if test_tokes[:5] == sp_train_con_tokes[:5]:
            com[count+1] = {}
            com[count+1][('test',k)] = sp_test_d[k]
            com[count+1][('sp_train',k2)] = sp_train_con_d[k2]
            count+=1
            found = True
            break
        elif test_tokes[-3:] == sp_train_con_tokes[-3:] and len(test_tokes) == len(sp_train_con_tokes):
            com[count+1] = {}
            com[count+1][('test',k)] = sp_test_d[k]
            com[count+1][('sp_train', k2)] = sp_train_con_d[k2]
            count+=1
            found = True
            break
    if found == False:
        for k2 in train_con_d:
            train_con_tokes = train_con_d[k2]['tweet_tokens']
            if test_tokes[:5] == train_con_tokes[:5]:
                com[count+1] = {}
                com[count+1][('test',k)] = sp_test_d[k]
                com[count+1][('train',k2)] = train_con_d[k2]
                count+=1
                found = True
                break
            elif test_tokes[-3:] == train_con_tokes[-3:] and len(test_tokes) == len(train_con_tokes):
                com[count+1] = {}
                com[count+1][('test',k)] = sp_test_d[k]
                com[count+1][('train', k2)] = train_con_d[k2]
                count+=1
                found = True
                break
    if found == False:
        for k2 in sp_dev_d:
            dev_tokes = sp_dev_d[k2]['tweet_tokens']
            if test_tokes[:5] == dev_tokes[:5]:
                com[count+1] = {}
                com[count+1][('test',k)] = sp_test_d[k]
                com[count+1][('dev', k2)] = sp_dev_d[k2]
                count+=1
                found = True
                break
            elif test_tokes[-3:] == dev_tokes[-3:] and len(test_tokes) == len(dev_tokes):
                com[count+1] = {}
                com[count+1][('test',k)] = sp_test_d[k]
                com[count+1][('dev', k2)] = sp_dev_d[k2]
                count+=1
                found = True
                break
        


# In[7]:


test_tweets = len(sp_test_d)
print('Number of tweets in test: {}'.format(test_tweets))


# In[8]:


test_founds = len(com)
print('Number of tweets in test: {}'.format(test_founds))


# In[9]:


# encoding differences are from the missing 1500 tweets?


# In[10]:


for k in list(com)[200:205]:
    print(com[k])


# In[14]:


# possible encoding fix ??


# In[16]:


# to restore proper emoji encoding
# https://stackoverflow.com/questions/20108312/how-can-i-restore-proper-encoding-of-4-byte-emoji-characters-that-have-been-stor


# In[ ]:




