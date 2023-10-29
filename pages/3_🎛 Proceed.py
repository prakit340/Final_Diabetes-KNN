import json
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Prakit Junkhum 633230010 24.2",
    page_icon= ":bar_chart:",
)

st.image('./images/header.jpg')
st.sidebar.markdown("# Proceed 🎛")

st.title("วิธีการดำเนินการ")
st.write("")
st.image('./images/overall.png')
st.write('''##### การพัฒนาการจำแนกโรคเบาหวานโดยการเรียนรู้ของเครื่อง ซึ่งได้นำทั้ง 3 เทคนิค มาทำการทดสอบเพื่อหาค่าความแม่นยำมากที่สุด เพื่อเลือกใช้อัลกอริทึมนั้นสำหรับเป็นประโยชน์ในการจำแนกโรคเบาหวาน เพื่อให้การใช้งานเป็นไปตามความต้องการ และสามารถให้ความแม่นยำได้สูงที่สุด โดยมีกระบวนการทำงานเบื้องต้นแบ่งเป็น 4 ขั้นตอนดังนี้ 1) รวบรวมข้อมูล และเตรียมข้อมูล 2) พัฒนาเลือกเทคนิคการจำแนกข้อมูล 3) การวิเคราะห์และประเมินผล และ 4) การทำให้พร้อมใช้งานผ่าน StreamLit 
    ''')

tab1, tab2, tab3, tab4 = st.tabs(["รวบวมข้อมูล เตรียมข้อมูล", "พัฒนาเลือกเทคนิคการจำแนกข้อมูล", "การวิเคราะห์ประเมินผล", "Deploy ผ่าน StreamLit"])
tab1.write("รวบวมข้อมูล เตรียมข้อมูล")
tab2.write("พัฒนาเลือกเทคนิคการจำแนกข้อมูล")
tab3.write("การวิเคราะห์ประเมินผล")
tab4.write("Deploy ผ่าน StreamLit")

# You can also use "with" notation:
with tab1:
    st.radio('Select one:', [1, 2])
with tab2:
    st.radio('Select one:', [1, 2])
with tab3:
    st.radio('Select one:', [1, 2])
with tab4:
    st.radio('Select one:', [1, 2])