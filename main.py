import streamlit as st
import pandas as pd
import plotly.express as px
import utils

st.set_page_config(layout="wide")

df = pd.read_csv("HIV_data 1990-2022.csv",index_col=0).dropna(axis=1)
country_list = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 
    'Central African Republic', 'Chad', 'Comoros', 'Congo', "Côte d'Ivoire", 
    'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 
    'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 
    'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 
    'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Sierra Leone', 'Somalia', 
    'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
]



l,r = st.columns([1,3])
country_name = l.selectbox("Country",sorted(set(country_list).intersection(df.columns)))
input_years = r.slider("Forecast Period (in Years)",1,10)

fit_df,pred_df = utils.predict(df,country_name,input_years)
plot_df = utils.combine(df[[country_name]],fit_df,pred_df).rename(columns={country_name:"New HIV Population"})

fig = px.line(plot_df,
              x=plot_df.index,
              y="New HIV Population",
              color="Tag",
              title=f"New HIV Population in {country_name}",
              color_discrete_sequence=["dodgerblue","mediumspringgreen","crimson"],
              height=600)
fig.update_layout(
    xaxis_title="Year", yaxis_title="New HIV Population"
)

r.plotly_chart(fig,use_container_width=True)
show_df = pred_df.copy().reset_index().rename(columns={"index":"Year",country_name:"New HIV Population"})
show_df["Year"] = show_df["Year"].astype("object")
show_df["New HIV Population"] = show_df["New HIV Population"].astype("int")
l.dataframe(show_df.style.format(thousands=''),use_container_width=True)

map_df = df[["2022"]].transpose().copy()
map_df["ISO"] = utils.country_iso_alpha3
st.dataframe(map_df)
