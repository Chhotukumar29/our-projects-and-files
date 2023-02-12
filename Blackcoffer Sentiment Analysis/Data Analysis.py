#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import os, shutil
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[3]:


flist = os.listdir(r'C:\Users\chhot\Dropbox\BlackCoffer\StopWords')


# In[4]:


with open ('StopWord.txt', 'w') as outfile:
    for file in flist:
        with open (file) as infile:
            for line in infile:
                line = line.split(' | ')[0]+'\n'
                outfile.write(line)


# In[5]:


df = pd.read_table('StopWords.txt', header=None, skip_blank_lines=True, encoding='windows-1252', names=['SW'])
stopwords = df['SW'].tolist()


# In[6]:


path = r"C:\Users\chhot\Dropbox\BlackCoffer\text files"
path1 = r"C:\Users\chhot\Dropbox\BlackCoffer\text file cleaned"

files = os.listdir(path)
for file in files:
    with open (r'{}\{}'.format(path,file), encoding='utf-8') as infile, open(r'{}\{}'.format(path1,file),'w',encoding='utf-8') as outfile:
        text = word_tokenize(infile.read())
        filetext = [txt for txt in text if not txt in stopwords]
        outfile.write(' '.join(filetext))


# In[8]:


path = r'C:\Users\chhot\Dropbox\BlackCoffer\MasterDictionary'
path1 = r'C:\Users\chhot\Dropbox\BlackCoffer\MasterDictionaryCleaned'

files = os.listdir(path)
for file in files:
    with open(r'{}\{}'.format(path,file), encoding='utf-8') as infile, open(r'{}\{}'.format(path1,file), 'w', encoding='utf-8') as outfile:
        filetext= [txt for txt in text if not txt in stopwords]
        outfile.write(' '.join(filetext))


# In[ ]:





# In[ ]:




