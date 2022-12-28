from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

app = Dash(__name__)

# get iris data from plotly
co2_df = pd.read_csv('clean/co2.csv')
ch4_df = pd.read_csv('clean/ch4.csv')

app.layout = html.Div(children=[
    html.H1(children='Co2 Emissions'),
    html.P(children='Select a country and compare the emissions of co2 and ch4'),
    dcc.Dropdown(
        id='dropdown_overview_co2',
        options=co2_df.columns,
        value=['Germany', 'United States', 'China'],
        multi=True
    ),
    dcc.Graph(id='graph_overview_co2'),
    html.P(children='Select a country and compare the emissions of co2 and ch4'),
    dcc.Dropdown(
        id='dropdown_overview_ch4',
        options=co2_df.columns,
        value=['Germany', 'United States', 'China'],
        multi=True),
    dcc.Graph(id='graph_overview_ch4')
])


@app.callback(
    Output('graph_overview_co2', 'figure'),
    Input('dropdown_overview_co2', 'value'))
def update_graph_overview_co2(dim):
    # get first column of df
    fig = px.scatter(co2_df, x=co2_df.columns[0], y=dim, )
    return fig


@app.callback(
    Output('graph_overview_ch4', 'figure'),
    Input('dropdown_overview_ch4', 'value'))
def update_graph_overview_ch4(dim):
    # get first column of df
    fig = px.scatter(ch4_df, x=ch4_df.columns[0], y=dim, )
    return fig


app.run_server(debug=True)
