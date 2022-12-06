import streamlit as st
import pandas as pd
import numpy as np
import nmslib
from gensim.models.fasttext import FastText
from time import time

df = pd.read_csv('Zevi_final_dataset.csv')
# print(df.head())

text_list = df['combined_clean_text'].tolist()
tok_text_list = [text.split() for text in text_list]

ft_model_trans = FastText.load('fasttext_vectors_trans.model') #load


# We are using Non-metric space library for first indexing and search
nmslib_index = nmslib.init(method='hnsw', space='cosinesimil')
nmslib_index.loadIndex("nmslib_index.bin")  # Loading the already trained index

def get_query_vector(text):
  text = text.lower().split()
  text_vector = [ft_model_trans[word] for word in text]
  text_vector = np.mean(text_vector, axis=0)
  return text_vector

def search(query, top_k):
    query_vector = get_query_vector(query)
    t0 = time()
    ids, distances = nmslib_index.knnQuery(query_vector, k=top_k)
    t1 = time()
    print(f'Searched {df.shape[0]} records in {round(t1-t0,4) } seconds \n')
    for i, j in zip(ids,distances):
        # search_score = (1-distances) + (0.1*df.popularity.values[i])/100
        st.write(df.objectID.values[i], df.text.values[i])

def main():
    st.title('Zevi search engine')
    st.markdown('''
    This is a semantic search engine where you type your query of products and number of 
    results you want to see in you want to see in your search and top results will be 
    shown. It can to some extent also handle bilingual hindi queries.
    ''')
    query = st.text_input(
        "Enter your query -"
    )
    top_k = st.number_input('Enter value of k (top k results will be shown)')

    if query:
        st.write("Top search results: ")
        search(query, int(top_k))

if __name__ == '__main__':
    main()

# st.markdown('''
# This is a dashboard showing the *average prices* of different types of :avocado:  
# Data source: [Kaggle](https://www.kaggle.com/datasets/timmate/avocado-prices-2020)
# ''')
