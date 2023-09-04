import streamlit as st
import pandas as pd
import plotly.express as px


df = pd.read_csv("HIV_data 1990-2022.csv",index_col=0).rename_axis("Country")
country_list = df.columns
country_name = st.selectbox("Country",country_list)
pdf = utils.predict(df,country_name,5)
fig = px.line(pdf,x=pdf.index,y="HIV Population",color="History")
st.plotly_chart(fig)