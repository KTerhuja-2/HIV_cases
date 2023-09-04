import streamlit as st
import pandas as pd
import plotly.express as px
import utils


df = pd.read_csv("HIV_data 1990-2022.csv",index_col=0).rename_axis("Country")
country_list = df.columns
country_name = st.selectbox("Country",country_list)
input_years = st.slider("Forecast Period",1,10)
plot_df = utils.combine(df[[country]],utils.predict(df,country_name,input_years))
fig = px.line(pdf,x=pdf.index,y=country,color="Tag")
st.plotly_chart(fig)