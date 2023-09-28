import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils
import predict_xgboost
import matplotlib.pyplot as plt
from darts import TimeSeries


st.set_page_config(page_title="HIV POC",page_icon="🎗️",layout="wide")
hide = """
<style>
MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}
</style>
"""
st.markdown(hide, unsafe_allow_html=True)
st.write(
    "<style>div.block-container{padding-top:0rem;}</style>", unsafe_allow_html=True
)
st.markdown("#### New HIV Cases Forecast in Africa")

df = pd.read_csv("HIV_data 1990-2022.csv",index_col=0).dropna(axis=1)
country_list = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 
    'Central African Republic', 'Chad', 'Comoros', 'Congo', "Côte d'Ivoire", 
    'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 
    'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 
    'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 
    'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Sierra Leone', 'Somalia', 
    'South Africa', 'South Sudan', 'Sudan', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
]



l,r = st.columns([1,4])
countries = sorted(set(country_list).intersection(df.columns))
country_name = l.selectbox("Country",countries,index=countries.index("South Africa"))
input_years = r.select_slider("Forecast Upto Year",[2023+i for i in range(10)])

#Linear Regression
fit_df,pred_df = utils.predict(df,country_name,input_years-2022)
# plot_df = utils.combine(df[[country_name]],fit_df,pred_df).rename(columns={country_name:"New HIV Population"})

#Exponential Smoothing
data = pd.read_csv('Cleaned HIV Data.csv')
series_aids = TimeSeries.from_dataframe(data,
                                    time_col="Time"
                                    )

series, prediction = utils.es_model(series_aids, country_name, input_years-2022)

# interactive_fig = plt.figure()
# series.plot(label= f"New HIV cases in {country_name}", color = 'green')
# prediction.plot(label=f"forecasted New HIV Cases in {country_name}", color = 'red')

# plt.legend(labelcolor = "white")


# fig = px.line(
#     plot_df,
#     x=plot_df.index,
#     y="New HIV Population",
#     color="Tag",
#     title=f"New HIV Population in {country_name}",
#     color_discrete_sequence=["dodgerblue","mediumspringgreen","crimson"],
#     height=600
#     )



# fig.update_layout(
#     xaxis_title="Year", yaxis_title="New HIV Population"
# )

actual_df =  series.pd_dataframe()
forecast_df = actual_df.tail(1).copy()
forecast_df = pd.concat([forecast_df,prediction.pd_dataframe()],axis=0)

forecast_df = (
    pd.concat(
        [
            forecast_df,
            actual_df,
        ],
        axis=1,
    )
    .set_axis(
        [
            f"New HIV cases in {country_name}",
            f"Forecasted New HIV Cases in {country_name}",
        ],
        axis=1,
    )
    .reset_index()
)
interactive_fig = px.line(
    forecast_df,
    x="Time",
    y=[
        f"New HIV cases in {country_name}",
        f"Forecasted New HIV Cases in {country_name}",
    ],
    color_discrete_sequence=["crimson","mediumseagreen"],
    markers=True,
)
interactive_fig.update_layout(
    title=f"New HIV Cases: 1990 - {input_years}",
    xaxis_title="Year",
    yaxis_title="New Cases",
    legend_title=" ",
    legend_orientation="h",
    legend_y=-0.1,
    legend_xanchor="center",
    legend_x=0.5,
    legend_traceorder="reversed",
)


with r:
    rl,rr = st.columns(2)
rl.plotly_chart(interactive_fig,use_container_width=True)
show_df = pred_df.copy().reset_index().rename(columns={"index":"Year",country_name:"New HIV Population"})
show_df["Year"] = show_df["Year"].astype("object")
show_df["New HIV Population"] = show_df["New HIV Population"].astype("int")
l.write(f"Forecast Upto year {input_years}")
l.dataframe(prediction.pd_dataframe().astype("int").style.format(thousands=''),use_container_width=True)


map_df = pd.read_csv("Forecasted HIV upto 2032.csv")
map_df.set_index("Time", inplace=True)
year = str(pd.to_datetime(str(input_years)))
year = year[:-9]
map_df = map_df.transpose()[[year]].copy()
map_df["ISO"] = utils.country_iso_alpha3
map_df.columns = ["New Cases", "ISO"]
fig2 = px.choropleth(
    map_df, 
    locations="ISO",
    color="New Cases",
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
# fig2.update_layout(plot_bgcolor = "rgb(14,17,23)")
rr.plotly_chart(fig2,use_container_width=True)





# #XGBoost
# forecasted = predict_xgboost.predict(country_name, input_years-2022)
# plot_values = forecasted.pd_dataframe()
# plot_values.reset_index(inplace=True)
# plot_values[country_name] = plot_values[country_name].astype('int')
# plot_values['Time'] = plot_values['Time'].astype('object')

# fig0 = px.line(
#     plot_values,
#     x="Time",
#     y=country_name

# )


