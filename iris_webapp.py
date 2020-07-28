# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import streamlit as st
import pickle



dec_tree_model=pickle.load(open('dec_tree_model.pkl','rb'))
knn_model=pickle.load(open('knn_model.pkl','rb'))
kmeans_model=pickle.load(open('kmeans_model.pkl','rb'))

def classify(num):
    if num<0.5:
        return 'Setosa'
    elif num <1.5:
        return 'Versicolor'
    else:
        return 'Virginica'
def main():
    st.title("IRIS")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Decision Tree','KNN','KMeans']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    sl=st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length', 0.0, 10.0)
    pw=st.slider('Select Petal Width', 0.0, 10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button('Classify'):
        if option=='Decision Tree':
            st.success(classify(dec_tree_model.predict(inputs)))
        elif option=='KNN':
            st.success(classify(knn_model.predict(inputs)))
        else: 
            st.success(classify(kmeans_model.predict(inputs)))    


if __name__=='__main__':
    main()


