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

st.image('./images/header.jpg')
st.sidebar.markdown("# Home 🏘︎")

st.write("# การจำแนกโรคเบาหวาน")
st.write("### จัดทำโดย")
st.write("# ประกฤต จันทร์ขำ 633230010 24.2")

with open('./files/wave.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.image('./images/diabetes.jpg')