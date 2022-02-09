#Adapted from code by Hank butler from GA lesson
import pickle
import streamlit as st

st.title('NoStupidQuestions or explainlikeimfive?')

# filepath to pickled model

with open('./models/gs_tfidf_log.pkl', 'rb') as pickle_in:
    pipe = pickle.load(pickle_in)

question = st.text_input('What is your question?', max_chars = 1000)

subreddit = pipe.predict([question])[0]
if subreddit == 0:
    subreddit = 'r/NoStupidQuestions'
else:
    subreddit = 'r/explainlikeimfive'
st.write(f'You should stick to {subreddit.title()}.')
