#Streamlit Application for Query Routing Using Artificial Intelligence

from keras import backend as K
from tensorflow.keras.models import Model, load_model
import streamlit as st
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(lower=True,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

#Model 

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model2.h5')
    return model

with st.spinner('Loading Model Into Memory....'):
    model = load_model()

labels = ['Credit Department', 'Mortgage Department', 'Debts Department','General Queries Department','Loan Department']    
    

#UI

#def main():
   
    
if __name__ == '__main__':
    st.title("")
    html_temp = """
    <div style="background-color:#7D0552;padding:10px">
    <h2 style="color:white;text-align:center;">Smart Customer Support</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Input the Query below:")
    sentence = st.text_area("Enter your Query/Issue here", height=200)
  # model, session = load_model()
    
    
#Button

predict_btt = st.button("Predict")
result=" "
    
    
if predict_btt:
    clean_text = []
    clean_text.append(sentence)
    sequences = tokenizer.texts_to_sequences(clean_text)
    MAX_SEQUENCE_LENGTH = 200
    data = pad_sequences(sequences, padding = 'post', maxlen =  MAX_SEQUENCE_LENGTH)
    result = model.predict(data)
    st.success("Entered query is being routed to {}.".format(labels[np.argmax(result)]))
