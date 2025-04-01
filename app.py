import pandas as pd
import pickle as pk
import streamlit as st

model=pk.load(open('spam.pkl','rb'))
cv=pk.load(open('cvspam.pkl','rb'))

st.header('Spam Detection App')

input_data = st.text_input('Enter Message Here')

if st.button('Predict'):
    input_data=cv.transform([input_data])
    prediction = model.predict(input_data)[0]
    st.markdown(prediction)