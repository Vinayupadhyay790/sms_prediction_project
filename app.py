import streamlit as st
import pickle
import pandas as pd
import numpy as np


import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
# libraries to transform input text
from nltk.corpus import stopwords


import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# to transform input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    
    stopword = stopwords.words('english')
    for i in text:
        if i not in stopword:
            y.append(i)            

    text = y[:]
    y.clear()
    
    import string
    punctuation = string.punctuation
    
    for i in text:
        if i not in punctuation:
            y.append(i)

    string = " ".join(y)
    
    text = y[:]
    y.clear() 
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

print(transform_text('Hii Vinay This side loving you'))


with open('vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl','rb') as f:
    model = pickle.load(f)

st.title('SMS Prection System')
st.markdown("<h4 style='font-size:24px;'>Drop your sms here to check if is spam or not....</h4>", unsafe_allow_html=True)

sms_text = st.text_area("",placeholder='Type something here....',height=100)

if st.button("Predict"):
    if sms_text=="":
        st.warning("Please enter some text.")
    else:
        transformed_text = transform_text(sms_text)
        vectorize_text = vectorizer.transform([transformed_text])
        prediction = model.predict(vectorize_text)[0]
        
        if prediction == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is NOT spam.")

st.markdown("---")
# st.markdown("Made with ❤️ using Streamlit"


# Second text area for feedback
st.markdown("<h4>Are you happy with the output? Write your feedback below.....<h5>",unsafe_allow_html=True)
feedback = st.text_area("", placeholder="Write your feedback here...",height=100)

st.button('Submit')