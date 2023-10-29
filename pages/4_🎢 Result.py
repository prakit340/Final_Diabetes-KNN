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

st.markdown("""<hr style="height:5px;border:none;color:#FF4B4B;background-color:#FF4B4B;" /> """, unsafe_allow_html=True)

st.header('Plot จากค่าของ Outcome ระหว่าง Diabetes กับ Normal ที่แต่ละ Attribute')
fig = sns.pairplot(df,hue='Outcome')
st.pyplot(fig)

st.markdown("""<hr style="height:5px;border:none;color:#FF4B4B;background-color:#FF4B4B;" /> """, unsafe_allow_html=True)

st.header("ศึกษาตัวแปรใดมีผลกับโรคเบาหวาน โดยใช้ตาราง Heatmap")
corr = df.corr()
st.write(corr.style.background_gradient(cmap='coolwarm'))

st.markdown("""<hr style="height:5px;border:none;color:#FF4B4B;background-color:#FF4B4B;" /> """, unsafe_allow_html=True)

x = df.iloc[:, 0:8]     # Select all row and column with 0-7
y = df['Outcome']       # Column Outcome for Output
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)  # Split data 80:20

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

dtModel = DecisionTreeClassifier(criterion='gini')    # Use Decision tree algorithm
dtModel.fit(x_train, y_train)                         # Train

nbModel = GaussianNB()                               # Use Naive Bayes algorithm
nbModel.fit(x_train, y_train)

knnModel = KNeighborsClassifier(n_neighbors=15, metric='euclidean', p=2)        # Use KNN algorithm
knnModel.fit(x_train, y_train)

################################ Predict

y_predDT = dtModel.predict(x_test)                  # Predict with x_test
accuracyDT = accuracy_score(y_test, y_predDT)

y_predNB = nbModel.predict(x_test)
accuracyNB = accuracy_score(y_test, y_predNB)

y_predKNN = knnModel.predict(x_test)
accuracyKNN = accuracy_score(y_test, y_predKNN)

st.header("ผลลัพธ์ Confusuin Matrix")
matrix = confusion_matrix(y_test, y_predKNN)
plt.figure(figsize=(10, 8))
fig2 = sns.heatmap(matrix, annot=True)
st.write(fig2)

st.markdown("""<hr style="height:5px;border:none;color:#FF4B4B;background-color:#FF4B4B;" /> """, unsafe_allow_html=True)

y_target = ['Diabetes', 'Normal']

st.write("ความแม่นยำของ Decision Tree Classifier:", '{:.2f}'.format(accuracyDT))
st.dataframe(
    pd.DataFrame(
        classification_report(y_test, y_predDT, target_names = y_target, output_dict=True)
    ).transpose()
)

st.write("ความแม่นยำของ Naive Bayes Tree Classifier:", '{:.2f}'.format(accuracyNB))
st.dataframe(
    pd.DataFrame(
        classification_report(y_test, y_predNB, target_names = y_target, output_dict=True)
    ).transpose()
)

st.write("ความแม่นยำของ K-Nearest Neighbors (KNN) Classifier:", '{:.2f}'.format(accuracyKNN))
st.dataframe(
    pd.DataFrame(
        classification_report(y_test, y_predKNN, target_names = y_target, output_dict=True)
    ).transpose()
)