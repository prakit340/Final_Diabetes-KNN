import json
import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

st.markdown(
    """
    <style>
    .reportview-container {
        background-color: gray;
    }
   </style>
    """,
    unsafe_allow_html=True
)
st.image('./images/header.jpg')

st.title('การจำแนกโรคเบาหวาน')
st.header('ประกฤต จันทร์ขำ 633230010 24.2')