import streamlit as st
import pandas as pd
import plotly.express as px
import utils


df = pd.read_csv("HIV_data 1990-2022.csv",index_col=0)
country_list = df.columns
country_name = st.selectbox("Country",country_list)
input_years = st.slider("Forecast Period",1,10)
st.dataframe(df[[country_name]])
st.dataframe(utils.predict(df,country_name,input_years))
plot_df = utils.combine(df[[country_name]],utils.predict(df,country_name,input_years))
st.dataframe(plot_df)
fig = px.line(plot_df,x=country_name,y=country_name,color="Tag")
st.plotly_chart(fig)