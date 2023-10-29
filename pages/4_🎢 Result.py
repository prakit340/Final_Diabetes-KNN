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

st.image('./images/header.jpg')
st.sidebar.markdown("# Result 🎢")

st.title("ผลการดำเนินงาน")
st.write("")

df=pd.read_csv('./data/diabetes.csv')

st.header('ตัวอย่างข้อมูล')
st.write(df.head(10))

st.header('Plot จากค่าของ Outcome ระหว่าง Diabetes กับ Normal ที่แต่ละ Attribute')
st.write(sns.PairGrid(df,hue='Outcome'))

st.header("ศึกษาตัวแปรใดมีผลกับโรคเบาหวาน โดยใช้ตาราง Heatmap")
corr = df.corr()
st.write(corr.style.background_gradient(cmap='coolwarm'))