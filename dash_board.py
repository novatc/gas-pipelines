from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# get iris data from plotly
co2_df = pd.read_csv('clean/co2.csv')
ch4_df = pd.read_csv('clean/ch4.csv')


def setup_layout():
    emission_app.layout = html.Div(children=intro() + overview_graph_c02() + overview_graph_ch4())


def intro():
    heading = html.H1(children='CO2 and CH4 Emissions Dashboard')
    intro_text = html.P(children='''This is a project to visualize the co2 emissions from around the world. The data 
    is from the [EDGAR](https://edgar.jrc.ec.europa.eu/dataset_ghg70). The data is from 2000-2021.''')
    table_of_content = html.Ul(children=[
        html.Li(children='Overview over all countries with both co2 and ch4 emissions'),
        html.Li(children='Histogram of the emissions'),
        html.Li(children='Regression of the emissions'),
        html.Li(children='Correlation of the emissions'),
        html.Li(children='Deep Learning Model to predict the emissions'),
    ])
    return [heading, intro_text, table_of_content]


def overview_graph_c02():
    heading_co2 = html.H2(
        children='Co2 Emissions over the years for each country')
    dropdown_co2 = html.Div([
        html.P(
            children='Select the countries you want to see'),
        dcc.Dropdown(
            id='dropdown_overview_co2',
            options=co2_df.columns,
            value=['Germany', 'United States', 'China'],
            multi=True
        )])

    graph_co2 = dcc.Graph(id='graph_overview_co2')

    return [heading_co2, dropdown_co2, graph_co2]

def overview_graph_ch4():
    heading_ch4 = html.H2(
        children='Ch4 Emissions over the years for each country')
    dropdown_ch4 = html.Div([
        html.P(
            children='Select the countries you want to see'),
        dcc.Dropdown(
            id='dropdown_overview_ch4',
            options=ch4_df.columns,
            value=['Germany', 'United States', 'China'],
            multi=True
        )])

    graph_ch4 = dcc.Graph(id='graph_overview_ch4')

    return [heading_ch4, dropdown_ch4, graph_ch4]


emission_app = Dash(__name__)
setup_layout()


@emission_app.callback(
    Output('graph_overview_co2', 'figure'),
    Input('dropdown_overview_co2', 'value'))
def update_graph_overview_co2(dim):
    # get first column of df
    fig = px.line(co2_df, x=co2_df.columns[0], y=dim, title='Co2 Emissions over the years for each country', labels={
                  'value': 'Co2 Emissions', 'variable': 'Country', 'year': 'Year', 'index': 'Year', 'co2': 'Co2 Emissions',
                    'ch4': 'Ch4 Emissions', 'Unnamed: 0': 'Year'})
    return fig
@emission_app.callback(
    Output('graph_overview_ch4', 'figure'),
    Input('dropdown_overview_ch4', 'value'))
def update_graph_overview_ch4(dim):
    # get first column of df
    fig = px.line(ch4_df, x=ch4_df.columns[0], y=dim, title='Ch4 Emissions over the years for each country', labels={
                  'value': 'Ch4 Emissions', 'variable': 'Country', 'year': 'Year', 'index': 'Year', 'co2': 'Co2 Emissions',
                    'ch4': 'Ch4 Emissions', 'Unnamed: 0': 'Year'})
    return fig


emission_app.run_server(debug=True)
