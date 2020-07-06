
import streamlit as st
import pickle
import sms1

import text_processing as tp

model=pickle.load(open('sms_spam\model1.pkl','rb'))

def main():
    user_input=st.text_area("ENTER THE TEXT")
    if st.button("submit"):
        user_input=tp.puncation(user_input)
        user_input=user_input.lower()
        user_input=tp.stopwordsremoval(user_input)
        user_input=tp.lemt(user_input)
        user_input=sms1.test(user_input)
        ans=model.predict(user_input)
        st.write(ans[0])
if __name__=='__main__':
    main()
