import json
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Prakit Junkhum 633230010 24.2",
    page_icon= ":bar_chart:",
)

st.image('./images/header.jpg')
st.sidebar.markdown("# Introduction 🏗︎")

st.title("บทนำ")
st.text("")
st.subheader('''โรคเบาหวานเป็นปัญหาสุขภาพที่ร้ายแรงส่งผลกระทบต่อผู้คนหลายล้านคนทั่วโลก 
         โรคเบาหวานประเภท 2 เป็นโรคที่พบบ่อยที่สุดชนิดหนึ่ง และสามารถทำให้เกิดปัญหาสุขภาพต่างๆ 
         เช่น โรคหัวใจ โรคหลอดเลือดสมอง และโรคไต ตามข้อมูลจากองค์การอนามัยโลก (WHO) 
         รายงานว่าจำนวนผู้ป่วยโรคเบาหวานเพิ่มขึ้นจาก 108 ล้านในปี 1980 เป็น 422 ล้านในปี 2014 และ 463 ล้านคนทั่วโลกในปี 2022 
         และคาดว่าจะเพิ่มขึ้นเป็น 642 ล้านคนภายในปี 2040 โรคเบาหวานประเภท 2 มักเกิดจากปัจจัยเสี่ยงหลายประการ 
         เช่น น้ำหนักเกิน หรืออ้วน ขาดการออกกำลังกาย ประวัติครอบครัว และภาวะแทรกซ้อนของการตั้งครรภ์ 
         ปัจจุบันยังไม่มีวิธีการรักษาโรคเบาหวานประเภทที่ 2 แต่สามารถควบคุมได้ด้วยการปรับเปลี่ยนวิถีชีวิต 
         เช่น การควบคุมอาหาร การออกกำลังกาย และการใช้ยา การศึกษานี้คือการพัฒนาโมเดลการทำนายโรคเบาหวานประเภท 2 
         โดยใช้อัลกอริทึม KNN โมเดลนี้สามารถใช้เพื่อระบุบุคคลที่มีความเสี่ยงสูงต่อโรคเบาหวานประเภทที่ 2 
         ซึ่งจะช่วยให้สามารถป้องกันหรือรักษาโรคได้ในระยะเริ่มต้น อัลกอริทึม KNN 
         เป็นอัลกอริทึมการเรียนรู้ของเครื่องแบบไม่เป็นเชิงเส้นที่ใช้ข้อมูลตัวอย่างเพื่อจำแนกข้อมูลใหม่ 
         อัลกอริทึ่มนี้จะเลือกข้อมูลตัวอย่างที่อยู่ใกล้กับข้อมูลใหม่ที่สุด k ครั้ง 
         และกำหนดคลาสของข้อมูลใหม่โดยพิจารณาจากคลาสของข้อมูลตัวอย่างเหล่านั้น''')
