#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


# In[2]:


flist = os.listdir(r'C:\Users\chhot\Dropbox\BlackCoffer\StopWords')

with open('StopWords.txt', 'w') as outfile:
    for file in flist:
        with open(file) as infile:
            for line in infile:
                line = line.split(' | ')[0]+'\n'
                outfile.write(line)


# In[3]:


df = pd.read_table('StopWords.txt', header=None, skip_blank_lines=True, encoding='windows-1252', names=['SW'])
StopWords = df['SW'].tolist()


# In[4]:


df = pd.read_excel('Output Data Structure.xlsx')
df


# In[5]:


def complex_word_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiou"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count += 1
    if count == 0:
        count += 1
    return count


# In[6]:


path1 = r'C:\Users\chhot\Dropbox\BlackCoffer\text files'
path2 = r"C:\Users\chhot\Dropbox\BlackCoffer\text file cleaned"

tfiles = os.listdir(path1)
wrdcnt = []
cwrdcnt = []
twac = []
for i in df['URL_ID'].values:
    with open(r'{}\{}.{}'.format(path1,i,'txt'),encoding='utf-8') as infile,open(r'{}\{}.{}'.format(path2,i,'txt'),'w',encoding='utf-8') as outfile:
        text = word_tokenize(infile.read())
        wrdcnt.append(len(text))
        cwcnt=0
        for wrd in text:
            if complex_word_count(wrd)>2:
                cwcnt+=1
        cwrdcnt.append(cwcnt)
        filtext = [txt for txt in text if not txt in StopWords]
        outfile.write(' '.join(filtext))
        twac.append(len(filtext))

df['wrdcnt'] = wrdcnt
df['COMPLEX WORD COUNT'] = cwrdcnt
df['twac'] = twac


# In[7]:


path3 = r"C:\Users\chhot\Dropbox\BlackCoffer\MasterDictionary" 
path4 = r"C:\Users\chhot\Dropbox\BlackCoffer\MasterDictionaryCleaned"

dfiles = os.listdir(path3)
for dfile in dfiles:
    with open(r'{}\{}'.format(path3,dfile), encoding='windows-1252') as infile, open(r'{}\{}'.format(path4,dfile), 'w', encoding='windows-1252') as outfile:
        text1 = word_tokenize(infile.read())
        filtext1 = [txt for txt in text1 if not txt in StopWords]
        outfile.write(' '.join(filtext1))


# In[8]:


with open(r'{}\{}'.format(path4,'positive-words.txt'), encoding='windows-1252') as file:
    PositiveWords = word_tokenize(file.read())

with open(r'{}\{}'.format(path4,'negative-words.txt'), encoding='windows-1252') as file:
    NegativeWords = word_tokenize(file.read())


# In[9]:


pscore = []
for i in df['URL_ID'].values:
    ps = 0
    with open(r'{}\{}.{}'.format(path2,i,'txt'), encoding='utf-8') as f:
        for txt in word_tokenize(f.read()):
            for txt1 in PositiveWords:
                if txt == txt1: ps+=1
    pscore.append(ps)
df['POSITIVE SCORE'] = pscore


# In[10]:


nscore = []
for i in df['URL_ID'].values:
    ns = 0
    with open(r'{}\{}.{}'.format(path2,i,'txt'), encoding='utf-8') as f:
        for txt in word_tokenize(f.read()):
            for txt1 in NegativeWords:
                if txt == txt1: ns-=1
    nscore.append(-ns)
df['NEGATIVE SCORE'] = nscore


# In[11]:


df['POLARITY SCORE']=round((df['POSITIVE SCORE']-df['NEGATIVE SCORE'])/((df['POSITIVE SCORE']+df['NEGATIVE SCORE'])+0.000001),2)
df['SUBJECTIVITY SCORE']=round((df['POSITIVE SCORE']+df['NEGATIVE SCORE'])/((df['twac'])+0.000001),2)


# In[12]:


sentcnt = []
for i in df['URL_ID'].values:
    with open(r'{}\{}.{}'.format(path1,i,'txt'),encoding='utf-8') as infile:
        sent = sent_tokenize(infile.read())
        sentcnt.append(len(sent))
        
df['sentcnt'] = sentcnt        
df['AVG SENTENCE LENGTH']=round(df['wrdcnt']/df['sentcnt'],2)
df['PERCENTAGE OF COMPLEX WORDS']=round(df['COMPLEX WORD COUNT']/df['wrdcnt'],2)
df['FOG INDEX']=round(0.4*(df['AVG SENTENCE LENGTH']+df['PERCENTAGE OF COMPLEX WORDS']),2)


# In[13]:


df['AVG NUMBER OF WORDS PER SENTENCE']=round(df['wrdcnt']/df['sentcnt'],2)


# In[14]:


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiou"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count += 1
    if word.endswith(('es','ed')):
        count -= 1
    if count == 0:
        count += 1
    return count


# In[15]:


NLTK_StopWords = list(set(stopwords.words('english')))

tfiles = os.listdir(path1)
WrdCnt = []
for i in df['URL_ID'].values:
    with open(r'{}\{}.{}'.format(path1,i,'txt'),encoding='utf-8') as f:
        text2 = word_tokenize(f.read())
        filtext2 = [txt for txt in text2 if not txt in NLTK_StopWords and txt not in ['?','!',',','.']]
        WrdCnt.append(len(filtext2))

df['WORD COUNT'] = WrdCnt


# In[16]:


scl = []
for i in df['URL_ID'].values:
    with open(r'{}\{}.{}'.format(path1,i,'txt'),encoding='utf-8') as f:
        words = word_tokenize(f.read())
        sc = 0
        for word in words:
            sc += syllable_count(word)
    scl.append(sc)

df['SYLLABLE PER WORD'] = scl/df['WORD COUNT'] 


# In[17]:


df.drop(columns=['twac','wrdcnt','sentcnt'], inplace=True)


# In[35]:


personal_nouns = []
personal_nouns =['I', 'we','my', 'ours','and' 'us','My','We','Ours','Us','And'] 
for i in df['URL_ID'].values:
    with open(r'{}\{}.{}'.format(path1,i,'txt'),encoding='utf-8') as f:
        words = word_tokenize(f.read())
        ans=0
        for word in words:
            if word in personal_nouns:
                ans+= 1
df['PERSONAL PRONOUNS'] = personal_nouns.append(ans)


# In[27]:


awl = []
for i in df['URL_ID'].values:
    with open(r'{}\{}.{}'.format(path1,i,'txt'),encoding='utf-8') as f:
        words = word_tokenize(f.read())
        avgwrdlgth = 0
        for word in words:
            arpitviansh = len(word)
            avgwrdlgth = avgwrdlgth + arpitviansh
    awl.append(avgwrdlgth)
df['AVG WORD LENGTH'] = (len(word))/(len(words))


# In[33]:


df.head(5)


# In[ ]:




