import joblib
import streamlit as st
import numpy as np

# with open('rand_model.pkl','rb') as f:
#     model = pickle.load(f)

model = joblib.load('rand_model.pkl')


st.title('STUDENT SCORE PREDICTION APP')
st.write('This app will predict the possible scores of students based on their performance')

st.subheader('Subject and Solutions for Prediction')
Subject = st.number_input('Enter subject value')
Solutions = st.number_input('Enter solutions value')


inputs = np.array([Subject,Solutions]).reshape(1, -1)


result = model.predict(inputs)

if st.button('Prediction'):
    st.write(f"Your score result is {result}")




