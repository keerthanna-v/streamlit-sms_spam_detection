
import streamlit as st
import pickle
import text_processing as tp
import warnings 
import numpy as np

warnings.filterwarnings('ignore')

model=pickle.load(open('model.pkl','rb'))
tf=pickle.load(open('tf.pkl','rb'))
 


def main():
    
    st.markdown("""<div style="background: linear-gradient(to right, #12c2e9, #c471ed, #f64f59);padding:10px">
                <h2 style="color:white;text-align:center;"> Sms Spam Classifier App </h2>
                </div>""",unsafe_allow_html=True)
    st.markdown("""   
                <br>
                <br>
                """,unsafe_allow_html=True)
    user_input=st.text_area("Enter The Message Text")
    st.write()
    if st.button("Submit"):
        if user_input:
            user_input=tp.puncation(user_input)
            user_input=user_input.lower()
            user_input=tp.stopwordsremoval(user_input)
            user_input=tp.lemt(user_input)
            user_input=tf.transform([user_input]).toarray()
            user_input=model.predict(user_input)
            if user_input[0]=="ham":
                st.success(user_input[0])
            else:
                st.warning(user_input[0])
        else:
            st.warning("Please Enter The Message Text")
if __name__=='__main__':
    main()