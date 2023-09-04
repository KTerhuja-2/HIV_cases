import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils

st.set_page_config(page_title="HIV POC",page_icon="🎗️",layout="wide")

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



l,r = st.columns([1,4])
country_name = l.selectbox("Country",sorted(set(country_list).intersection(df.columns)))
input_years = r.select_slider("Forecast Upto Year",[2023+i for i in range(10)])

fit_df,pred_df = utils.predict(df,country_name,input_years-2022)
plot_df = utils.combine(df[[country_name]],fit_df,pred_df).rename(columns={country_name:"New HIV Population"})

fig = px.line(
    plot_df,
    x=plot_df.index,
    y="New HIV Population",
    color="Tag",
    title=f"New HIV Population in {country_name}",
    color_discrete_sequence=["dodgerblue","mediumspringgreen","crimson"],
    height=600
    )
fig.update_layout(
    xaxis_title="Year", yaxis_title="New HIV Population"
)
with r:
    rl,rr = st.columns(2)
rl.plotly_chart(fig,use_container_width=True)
show_df = pred_df.copy().reset_index().rename(columns={"index":"Year",country_name:"New HIV Population"})
show_df["Year"] = show_df["Year"].astype("object")
show_df["New HIV Population"] = show_df["New HIV Population"].astype("int")
l.write(f"Forecast Upto year {input_years}")
l.dataframe(show_df.style.format(thousands=''),use_container_width=True)


map_df = pd.read_csv("HIV_data 1990-2032.csv",index_col=0).dropna(axis=1)
map_df = map_df[sorted(set(country_list).intersection(df.columns))].transpose()[[input_years]].copy()
map_df["ISO"] = utils.country_iso_alpha3
fig2 = px.choropleth(
    map_df, 
    locations="ISO",
    color=input_years,
    hover_name=map_df.index,
    color_continuous_scale=px.colors.sequential.YlOrRd,
    # color_continuous_scale=px.colors.diverging.balance,
    scope="africa",
    title=f"New HIV Population in year {input_years}",
    height=600
    )
fig2.update_geos(
    bgcolor="rgb(14,17,23)",
    showcoastlines=False,
    showland=False,
    showocean=False,
    showlakes=False,
    showrivers=False,
    showframe=True,
    framewidth=5,
    framecolor="rgb(150,150,150)",
    )
fig.update_layout(plot_bgcolor = "rgb(14,17,23)")
rr.plotly_chart(fig2,use_container_width=True)
