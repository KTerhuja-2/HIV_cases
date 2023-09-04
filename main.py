import streamlit as st
import pandas as pd


df = pd.read_csv('New HIV infections_New HIV infections - All ages_Population_ All ages.csv')
country_list = df["Country"]
country_name = st.selectbox("Country",country_list)

st.dataframe(df[df["Country"]==country_name])