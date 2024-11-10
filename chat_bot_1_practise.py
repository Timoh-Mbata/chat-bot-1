#!/usr/bin/env python
# coding: utf-8

# In[11]:


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the fashion text file
with open('fashion.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Preprocess each sentence (removes stopwords, punctuation, and lemmatizes)
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)  # Return as a preprocessed string, not a list of words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to calculate similarity between user query and sentences in corpus
def get_most_relevant_sentence(query):
    # Preprocess the query
    processed_query = preprocess(query)
    
    # Create TF-IDF vectorizer and fit the corpus + query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [processed_query])
    
    # Calculate cosine similarity between the query and each sentence
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Get the index of the most similar sentence
    most_similar_idx = cosine_sim.argmax()
    
    return sentences[most_similar_idx]

# Define the chatbot function
def chatbot(query):
    # Get the most relevant sentence from the corpus
    response = get_most_relevant_sentence(query)
    return response

# Define the Streamlit app function
def main():

    st.title("Chatbot")

    st.write("Hello! I'm a chatbot. Ask me anything about fashion!")

    # Get the user's question
    question = st.text_input("You:")

    # Create a button to submit the question
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()

# The ‘nltk’ library is used for natural language processing tasks such as tokenization, lemmatization, and stopword removal. The ‘string’ library is used for string operations. The ‘streamlit’ library is used to create the web-based chatbot interface.


# The ‘nltk.download()’ function is used to download additional resources needed for the nltk library. In this case, we are downloading the punkt and averaged_perceptron_tagger resources. These resources are needed for tokenization and part-of-speech tagging tasks.


# Once you have imported the necessary libraries, you can use their functions and classes to perform various NLP tasks and create your chatbot.


# In[12]:


get_ipython().system('jupyter nbconvert --to python chat bot 1 practise.ipynb')


# In[ ]:




