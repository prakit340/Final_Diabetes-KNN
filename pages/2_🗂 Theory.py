import json
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Prakit Junkhum 633230010 24.2",
    page_icon= ":bar_chart:",
)

st.image('./images/header.jpg')
st.sidebar.markdown("# Theory 🗂")

st.title("ทฤษฎี")
st.write("")
st.write('''โรคเบาหวานเป็นปัญหาสุขภาพที่ร้ายแรงส่งผลกระทบต่อผู้คนหลายล้านคนทั่วโลก 
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

st.image('./images/diabetes.jpg')

col1, col2, col3, col4 = st.columns(4)
col1.write('โรคเบาหวานชนิดที่ 2')
col2.write('การจำแนกด้วยวิธี KNN')
col3.write('การจำแนกด้วยวิธี Naïve Bayes')
col4.write('การจำแนกด้วยวิธี Decision Tree')

# Three columns with different widths
col1, col2, col3, col4 = st.columns([1,1,1,1])
# col1 is wider

# Using 'with' notation:
with col1:
    st.write('''เบาหวานชนิดที่ 2 เป็นโรคเรื้อรังที่เกิดจากร่างกายไม่สามารถผลิตอินซูลินได้เพียงพอ หรือไม่สามารถตอบสนองต่ออินซูลินได้อย่างมีประสิทธิภาพ 
    อินซูลินเป็นฮอร์โมนที่ช่วยให้น้ำตาลเข้าสู่เซลล์เพื่อให้ร่างกายน้ำไปใช้เป็นแหล่งพลังงาน เมื่อร่างกายไม่สามารถผลิตอินซูลินได้เพียงพอหรือไม่สามารถตอบสนองต่ออินซูลินได้อย่างมีประสิทธิภาพ 
    น้ำตาลจะสะสมในเลือด ซึ่งอาจนำไปสู่ปัญหาสุขภาพต่างๆ เช่น โรคหัวใจ โรคหลอดเลือดสมอง โรคไต และโรคตา อาการของโรคเบาหวานชนิดที่ 2 มักไม่ปรากฏชัดเจนในช่วงแรก 
	อาการที่พบบ่อยได้แก่ ปัสสาวะบ่อย กระหายน้ำบ่อย น้ำหนักลดโดยไม่ทราบสาเหตุ เหนื่อยล้า มองเห็นภาพซ้อน แผลหายช้า 
	ปัจจัยที่เสี่ยงในการเป็นโรคเบาหวานชนิดที่ 2 ได้แก่ อายุ น้ำหนักเกินหรือโรคอ้วน ประวัติครอบครัว ภาวะแทรกซ้อนขณะตั้งครรภ์ ประวัติโรคเบาหวานชนิดที่ 2 
	การรักษาโรคเบาหวานชนิดที่ 2 มีวัตถุประสงค์เพื่อควบคุมระดับน้ำตาลในเลือดให้อยู่ในระดับปกติ การรักษาประกอบด้วย การควบคุมอาหาร การออกกำลังกาย และการใช้ยา 
	ภาวะแทรกซ้อนจากโรคเบาหวานชนิดที่ 2 ได้แก่ โรคหัวใจ โรคหลอดเลือดสมอง โรคไต โรคตา โรคประสาท
	การป้องกันโรคเบาหวานชนิดที่ 2 ทำได้โดยการ ควบคุมน้ำหนัก ออกกำลังกายเป็นประจำ รับประทานอาหารที่มีกากใยสูง ไขมันต่ำ และโปรตีนปานกลาง ตรวจสุขภาพเป็นประจำ 
    ''')
with col2:
    st.write('''K-NN (K-nearest Neighbor) คือ วิธีการหาสมาชิกที่มีความคล้ายคลึงกันมากที่สุด โดยการหาค่าระยะห่างระหว่างข้อมูล (Standard Euclidean Distance) [3] เป็นอัลกอริทึมหนึ่งที่สามารถใช้งานได้ดีเพื่อเติมเต็มข้อมูลที่ขาดหายไป ใช้หลักการวัดระยะห่าง ระหว่างข้อมูลขาดหายกับข้อมูลแต่ละตัว แล้วประมาณค่าที่ขาดหายไปด้วยค่าเฉลี่ยของข้อมูลที่ใกล้ที่สุดจำนวน k ตัว นิยมใช้การคำนวณระยะทางแบบยูคลิเดียน (Euclidean Distance) นั่นคือ ถ้า d เป็นระยะห่างระหว่างจุด (x1, y1) และ (x2, y2) จะได้ ''')
    st.image('./images/knn.jpg')