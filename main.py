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

ft = """
<style>
a:link , a:visited{
color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: #0283C3; /* theme's primary color*/
background-color: transparent;
text-decoration: underline;
}

#page-container {
  position: relative;
  min-height: 10vh;
}

footer{
    visibility:hidden;
}

.footer {
position: relative;
left: 0;
top:0;
bottom: 0;
width: 100%;
background-color: transparent;
color: #808080; /* theme's text color hex code at 50 percent brightness*/
text-align: left; /* you can replace 'left' with 'center' or 'right' if you want*/
}
</style>

<div id="page-container">

<div class="footer">
<p style='font-size: 0.875em;'>Developed by <a style='display: inline; text-align: left;' href="https://www.virtusa.com/" target="_blank">Virtusa <img src = "data:image/jpeg;base64,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" height = "15"> </a><br 'style= top:3px;'> 
Data & Analytics Team  </a></p>
</div>

</div>
"""
st.write(ft, unsafe_allow_html=True)



