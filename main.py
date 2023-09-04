import streamlit as st
import pandas as pd
import plotly.express as px
import utils

df = pd.read_csv("HIV_data 1990-2022.csv",index_col=0)
country_list = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 
    'Central African Republic', 'Chad', 'Comoros', 'Congo', "Côte d'Ivoire", 
    'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 
    'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 
    'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 
    'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Sierra Leone', 'Somalia', 
    'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
]
country_name = st.selectbox("Country",country_list)
input_years = st.slider("Forecast Period",1,10)
# st.dataframe(df[[country_name]])
# st.dataframe(utils.predict(df,country_name,input_years))
plot_df = utils.combine(df[[country_name]],utils.predict(df,country_name,input_years)).rename(columns={country_name:"HIV Population"})
# st.dataframe(plot_df)
fig = px.line(plot_df,x=plot_df.index,y="HIV Population",color="Tag",title=f"HIV Population in {country_name}")
fig.update_layout(
    xaxis_title="Year", yaxis_title="HIV Population"
)
st.plotly_chart(fig)

