# Load libraries
import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier          # Import Decision Tree Classifier
from sklearn.naive_bayes import GaussianNB              # Import GaussianNB
from sklearn.neighbors import KNeighborsClassifier       # Import KNN
from sklearn.model_selection import train_test_split    # Import train_test_split function
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score    # Import measure performance function
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="Prakit Junkhum 633230010 24.2",
    page_icon= ":bar_chart:",
)

with open('./files/wave.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.image('./images/header.jpg')
st.sidebar.markdown("# Try it out 🗓︎")

st.title("Diabetes Prediction")
st.header("Diabetes Prediction Prakit Junkhum 633230010 24.2")

df=pd.read_csv('./data/diabetes.csv')
st.write(df.head(12))

#st.line_chart(df)
#st.line_chart(df, x="interest_rate", y="unemployment_rate", color="stock_index_price")
# st.line_chart(
#    df, x="interest_rate", y=["unemployment_rate", "stock_index_price"], color=["#FF0000", "#0000FF"]  # Optional
# )

x = df.iloc[:, 0:8]     # Select all row and column with 0-7
y = df['Outcome']       # Column Outcome for Output

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)  # Split data 80:20

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

knnModel = KNeighborsClassifier(n_neighbors=15, metric='euclidean', p=2)        # Use KNN algorithm
knnModel.fit(x_train, y_train)

x1=st.number_input("กรุณาป้อนข้อมูล Pregnancies:")
x2=st.number_input("กรุณาป้อนข้อมูล Glucose:")
x3=st.number_input("กรุณาป้อนข้อมูล BloodPressure:")
x4=st.number_input("กรุณาป้อนข้อมูล SkinThickness:")
x5=st.number_input("กรุณาป้อนข้อมูล Insulin:")
x6=st.number_input("กรุณาป้อนข้อมูล BMI:")
x7=st.number_input("กรุณาป้อนข้อมูล DiabetesPedigreeFunction:")
x8=st.number_input("กรุณาป้อนข้อมูล Age:")

if st.button("พยากรณ์ข้อมูล"):
    x_input=[[x1,x2,x3,x4,x5,x6,x7,x8]]
    y_predict=knnModel.predict(scaler.transform(x_input))
    if y_predict == 1:
        st.write("มีความเสียงเป็นเบาหวาน")
    else:
        st.write("ไม่มีความเสี่ยง / ปกติ")
    st.button("ไม่พยากรณ์ข้อมูล")
else:
    st.button("ไม่พยากรณ์ข้อมูล")