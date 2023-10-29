import json
import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

st.set_page_config(
    page_title="Prakit Junkhum 633230010 24.2",
    page_icon= ":bar_chart:",
)

st.markdown(
    """
    <style>
    body {
        background-color: gray;
    }
   </style>
    """,
    unsafe_allow_html=True
)

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

st.image('./images/header.jpg')

st.title('การจำแนกโรคเบาหวาน')
st.header('ประกฤต จันทร์ขำ 633230010 24.2')