
# coding: utf-8

# # Tokenization

# In[1]:


import nltk


# In[5]:


nltk.download('wordnet')


# In[2]:


text= "This is Andrew's text, isn't it?"
tokenizer=nltk.tokenize.WhitespaceTokenizer()
tokenizer.tokenize(text)


# In[3]:


tokenizer=nltk.tokenize.TreebankWordTokenizer()
tokenizer.tokenize(text)


# In[4]:


tokenizer=nltk.tokenize.WordPunctTokenizer()
tokenizer.tokenize(text)



# # Stemming

# In[6]:


text = "feet wolves cats talked"
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)


# In[7]:


stemmer = nltk.stem.PorterStemmer()
" ".join(stemmer.stem(token) for token in tokens)


# In[8]:


stemmer = nltk.stem.WordNetLemmatizer()
" ".join(stemmer.lemmatize(token) for token in tokens)

