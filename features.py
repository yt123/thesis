#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import AUT
from google.cloud import translate
from nltk.corpus import wordnet as wn
import os


# In[2]:


df1 = AUT.read_csv("CES_bp2016_brick_scored_rater01.csv")
df1['rater'] = 1

df2 = AUT.read_csv("CES_bp2016_brick_scored_rater02.csv")
df2['rater'] = 2

df3 = AUT.read_csv("CES_bp2017_brick_scored_rater01.csv")
df3['rater'] = 3

df4 = AUT.read_csv("CES_bp2017_brick_scored_rater02.csv")
df4['rater'] = 1

df5 = AUT.read_csv("CES_tz2016_brick_scored_rater01.csv")
df5['rater'] = 1

df6 = AUT.read_csv("CES_tz2016_brick_scored_rater02.csv")
df6['rater'] = 2

df7 = AUT.read_csv("CES_tz2017_brick_scored_rater01.csv")
df7['rater'] = 3

df8 = AUT.read_csv("CES_tz2017_brick_scored_rater02.csv")
df8['rater'] = 2


# In[4]:


df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
df = df.dropna()


# In[5]:


sent_vec = AUT.fasttext_vectors(df)


# In[6]:


df = AUT.append_fasttext_vectors(df, sent_vec)


# In[ ]:


df = AUT.semantic_distance(df, sent_vec)


# In[10]:


a = pd.DataFrame(dist_matrix)
dist = a.iloc[1:,0].reset_index(drop = True).to_frame(name = 'dist')


# In[11]:


b = a.iloc[1:, 1:]
b.index = range(len(b.index))
b.columns = range(b.shape[1])


# In[12]:


# # select the pairs with semantic distance < .2
all_pairs = b.where(b < .2).stack().index.values

# # exclude the equal pairs that are of the same indexes (e.g., (1, 1))
unequal_pairs = [(min(p0,p1), max(p0,p1)) for (p0, p1) in all_pairs if p0 != p1]
unequal_pairs = sorted(set(unequal_pairs))


# In[ ]:


for p0, p1 in unequal_pairs:
     if df.loc[p1, 'cleaned_response'] != df.loc[p0, 'cleaned_response']:
         df.loc[p1, 'cleaned_response'] = df.loc[p0, 'cleaned_response']


# In[7]:


df = AUT.count_frequencies(df)


# In[8]:


df['1/freq'] = 1/df['freq']


# In[ ]:


from google.cloud import translate

# Instantiates a client
translate_client = translate.Client()


# In[ ]:


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "translate-f481dadfeb8d.json


# In[ ]:


def translate_en(df):
    orig_responses = list(df['cleaned_response'])
    results = []
    translate_client = translate.Client()
    for begin in range(0, len(orig_responses), 100):
        end = min(begin + 100, len(orig_responses))
        temp_results = translate_client.translate(orig_responses[begin:end], target_language='en')
        results.extend(result['translatedText'] for result in temp_results)
    df['response_en'] = results
    return df


# In[ ]:


df.to_csv('test.csv')


# In[ ]:


df = translate_en(df)


# In[ ]:


df_mean = df[['cleaned_response', 'originality']]
df_mean = df_mean.groupby('cleaned_response').mean()
df_mean = df_mean.rename({'originality': 'mean'}, axis='columns')
df = df.merge(df_mean, on='cleaned_response')


# In[ ]:


from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
 
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def wup_similarity(synset1, synset2):
    return synset1.wup_similarity(synset2)

def path_similarity(synset1, synset2):
    return synset1.path_similarity(synset2)

def sentence_similarity(sentence1, sentence2, similarity_func=wup_similarity):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
        
    score, count = 0.0, 0
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([similarity_func(synset, ss) or -1 for ss in synsets2], default=-1)

        # Check that the similarity could have been computed
        if best_score >= 0:
            score += best_score
            count += 1
 
    # Average the values
    if count > 0:
        return score / count
    return 0

def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2 


# In[ ]:


sentences = df['response_en']
 
focus_sentence = "brick"

df['wup_similarity'] = [sentence_similarity(focus_sentence, sentence) for sentence in sentences]
df['path_similarity'] = [sentence_similarity(focus_sentence, sentence, path_similarity) for sentence in sentences]
#df['lch_similarity'] = [sentence_similarity(focus_sentence, sentence, lch_similarity) for sentence in sentences]


# In[ ]:


AUT.save_csv(df)

