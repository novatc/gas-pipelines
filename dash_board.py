from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from data_preprocess import get_country_from_data, get_country_from_data_with_index, \
    get_country_from_ch4_data_with_index, get_population_data, merge_dataframes, get_countries_from_data
from forcasting_methodes.forcasting import SARIMA_co2
from forcasting_methodes.lr import linear_regression, linear_regression_co2_dash, linear_regression_ch4_dash
from population.population_calculation import population_and_co2, population_and_ch4

# get iris data from plotly
co2_df = pd.read_csv('clean/co2.csv')
ch4_df = pd.read_csv('clean/ch4.csv')
population_df = get_population_data()


def setup_layout():
    emission_app.layout = html.Div(
        children=intro() + overview_graph_c02() +
                 overview_graph_ch4() +
                 simple_correlation() +
                 clustering_co2() +
                 histogram_co2()
                 + histogram_ch4()
                 + population_correlation_co2()
                 + population_correlation_ch4() +
                 linear_regression_graph() +
                 linear_regression_graph_ch4()
    )


def intro():
    heading = html.H1(children='CO2 and CH4 Emissions Dashboard')
    intro_text = html.P(children=['''This is a project to visualize the co2 emissions from around the world. The data 
    is from the EDGAR( https://edgar.jrc.ec.europa.eu/dataset_ghg70). The data is from 2000-2021.''',
                                  html.Br(),
                                  ])
    description = html.P(children=['''This project aims to awnser the following questions:''',
                                   html.Br(),
                                   html.Ul(children=[
                                       html.Li(
                                           children='Is there a correlation between the co2 emissions and CH4 emissions?'),
                                       html.Li(
                                           children='Is there a correlation between the co2 emissions and the population for specific countries?'),
                                       html.Li(
                                           children='Is it possible to predict the emissions with linear regression?'),
                                   ])])
    table_of_content_text = html.P(children=['''This dashboard contains the following sections:'''])

    table_of_content = html.Ul(children=[
        html.Li(children='Overview over all countries with both co2 and ch4 emissions'),
        html.Li(children='Clustering of the emissions'),
        html.Li(children='Histogram of the emissions'),
        html.Li(children='Correlation of the emissions'),
        html.Li(children='Linear regression to predict the emissions'),
    ])
    return [heading, intro_text, description, table_of_content_text, table_of_content]


def overview_graph_c02():
    heading_co2 = html.H2(
        children='Co2 Emissions over the years for each country')
    dropdown_co2 = html.Div([
        html.P(
            children='Select the countries you want to see the emissions for'),
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
        children='Ch4 Emissions over the years for each country the emissions for. You can select multiple countries and compare the emissions.')
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


def histogram_co2():
    heading_ch4 = html.H2(
        children='CO2 Emissions over the years for each country ')
    dropdown_ch4 = html.Div([
        html.P(
            children='Ch4 Emissions over the years for each country the emissions for. You can select multiple countries and compare the emissions.'),

        dcc.Dropdown(
            id='dropdown_histogram_co2',
            options=ch4_df.columns,
            value=['Germany', 'United States', 'China'],
            multi=True
        )])

    graph_ch4 = dcc.Graph(id='graph_histogram_co2')

    return [heading_ch4, dropdown_ch4, graph_ch4]


def histogram_ch4():
    heading_ch4 = html.H2(
        children='CH4 Emissions over the years for each country')
    dropdown_ch4 = html.Div([
        html.P(
            children='Select the countries you want to see represented as a histogram. You can select multiple '
                     'countries and compare them.'),
        dcc.Dropdown(
            id='dropdown_histogram_ch4',
            options=ch4_df.columns,
            value=['Germany', 'United States', 'China'],
            multi=True
        )])

    graph_ch4 = dcc.Graph(id='graph_histogram_ch4')

    return [heading_ch4, dropdown_ch4, graph_ch4]


def simple_correlation():
    heading_correlation = html.H2(children='Correlation between the emissions')
    dropdown_correlation = html.Div([
        html.P(
            children='Select the country you want to see the correlation between the co2 and ch4 emissions'),
        dcc.Dropdown(
            id='dropdown_correlation',
            options=co2_df.columns,
            value='Germany',
            multi=False
        ),
    ])

    graph_correlation = dcc.Graph(id='graph_correlation')

    return [heading_correlation, dropdown_correlation, graph_correlation]


def linear_regression_graph():
    heading_regression = html.H2(children='Linear Regression for Co2 Emissions')
    dropdown_regression = html.Div([
        html.P(
            children='Select the countries you want to see being predicted. A linear regression system is used to predict the next 2 years.'),
        dcc.Dropdown(
            id='dropdown_regression_co2',
            options=co2_df.columns,
            value='Germany',
            multi=False
        ),
    ])

    graph_regression = dcc.Graph(id='graph_regression_co2')

    return [heading_regression, dropdown_regression, graph_regression]


def linear_regression_graph_ch4():
    heading_regression = html.H2(children='Linear Regression for Ch4 Emissions')
    dropdown_regression = html.Div([
        html.P(
            children='Select the countries you want to see being predicted. A linear regression system is used to predict the next 2 years.'),
        dcc.Dropdown(
            id='dropdown_regression_ch4',
            options=ch4_df.columns,
            value='Germany',
            multi=False
        ),
    ])

    graph_regression = dcc.Graph(id='graph_regression_ch4')

    return [heading_regression, dropdown_regression, graph_regression]


def population_correlation_co2():
    heading_regression = html.H2(children='Population and Co2 Emissions')
    dropdown_regression = html.Div([
        html.P(
            children='Select the countries you want to see the correlation between the emissions and the population'),
        dcc.Dropdown(
            id='dropdown_correlation_co2',
            options=ch4_df.columns,
            value='Germany',
            multi=False
        )])

    graph_regression = dcc.Graph(id='graph_correlation_co2')

    return [heading_regression, dropdown_regression, graph_regression]


def population_correlation_ch4():
    heading_regression = html.H2(children='Population and Ch4 Emissions')
    dropdown_regression = html.Div([
        html.P(
            children='Select the countries you want to see the correlation between the emissions and the population'),
        dcc.Dropdown(
            id='dropdown_correlation_ch4',
            options=ch4_df.columns,
            value='Germany',
            multi=False
        )])

    graph_regression = dcc.Graph(id='graph_correlation_ch4')

    return [heading_regression, dropdown_regression, graph_regression]


def clustering_co2():
    heading_clustering = html.H2(children='Clustering of Co2 Emissions')
    dropdown_clustering = html.Div([
        html.P(
            children='Here are all the countries clustered by their emissions. You can select which emissions you want '
                     'to see. This should help you to get an understanding of different country groups'),
        dcc.Dropdown(
            id='dropdown_clustering_co2',
            options=['Co2', 'CH4'],
            value='Co2',
            multi=False
        )])
    graph_clustering_co2 = dcc.Graph(id='graph_clustering_co2')

    return [heading_clustering, dropdown_clustering, graph_clustering_co2]


emission_app = Dash(__name__)
setup_layout()


@emission_app.callback(
    Output('graph_overview_co2', 'figure'),
    Input('dropdown_overview_co2', 'value'))
def update_graph_overview_co2(dim_co2):
    # get first column of df
    fig = px.line(co2_df, x=co2_df.columns[0], y=dim_co2, title='Co2 Emissions over the years for each country',
                  labels={
                      'value': 'Co2 Emissions', 'variable': 'Country', 'year': 'Year', 'index': 'Year',
                      'co2': 'Co2 Emissions',
                      'ch4': 'Ch4 Emissions', 'Unnamed: 0': 'Year'})
    return fig


@emission_app.callback(
    Output('graph_overview_ch4', 'figure'),
    Input('dropdown_overview_ch4', 'value'))
def update_graph_overview_ch4(dim_ch4):
    # get first column of df
    fig = px.line(ch4_df, x=ch4_df.columns[0], y=dim_ch4, title='Ch4 Emissions over the years for each country',
                  labels={
                      'value': 'Ch4 Emissions', 'variable': 'Country', 'year': 'Year', 'index': 'Year',
                      'co2': 'Co2 Emissions',
                      'ch4': 'Ch4 Emissions', 'Unnamed: 0': 'Year'})
    return fig


@emission_app.callback(
    Output('graph_correlation', 'figure'),
    Input('dropdown_correlation', 'value'))
def update_graph_correlation(country_correlation):
    corr_co2 = pd.read_csv('clean/co2.csv')
    corr_ch4 = pd.read_csv('clean/ch4.csv')
    combined_correlation = merge_dataframes(corr_co2, corr_ch4)
    selected_country_correlation = get_country_from_data(combined_correlation, country_correlation)
    print(selected_country_correlation.columns)
    fig = px.scatter(selected_country_correlation, x='co2', y='ch4', trendline='ols')
    return fig


@emission_app.callback(
    Output('graph_regression_co2', 'figure'),
    Input('dropdown_regression_co2', 'value'))
def update_graph_regression_co2(country_co2):
    country_df = get_country_from_data_with_index(co2_df, country_co2)
    result_df = linear_regression_co2_dash(country_df, 10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df['time'], y=result_df['co2'], mode='lines', name='Actual'))

    # Add a scatter plot for the predicted values
    fig.add_trace(
        go.Scatter(x=result_df.iloc[-51:-1]['time'], y=result_df.iloc[-51:-1]['co2'], mode='lines', name='Predicted'))

    # Update the layout of the plot
    fig.update_layout(title='CO2 Emissions Over Time', xaxis_title='Date', yaxis_title='CO2 Emissions')

    return fig


@emission_app.callback(
    Output('graph_regression_ch4', 'figure'),
    Input('dropdown_regression_ch4', 'value'))
def update_graph_regression_ch4(country_ch4):
    country_df = get_country_from_ch4_data_with_index(ch4_df, country_ch4)
    result_df = linear_regression_ch4_dash(country_df, 10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df['time'], y=result_df['ch4'], mode='lines', name='Actual'))

    # Add a scatter plot for the predicted values
    fig.add_trace(
        go.Scatter(x=result_df.iloc[-51:-1]['time'], y=result_df.iloc[-51:-1]['ch4'], mode='lines', name='Predicted'))

    # Update the layout of the plot
    fig.update_layout(title='CH4 Emissions Over Time', xaxis_title='Date', yaxis_title='CH4 Emissions')

    return fig


@emission_app.callback(
    Output('graph_histogram_co2', 'figure'),
    Input('dropdown_histogram_co2', 'value'))
def update_graph_histogram_co2(country_histogram_co2):
    # get first column of df
    fig = px.histogram(co2_df, x=country_histogram_co2, title='Co2 Emissions over the years for each country', labels={
        'value': 'Co2 Emissions', 'variable': 'Country', 'year': 'Year', 'index': 'Year', 'co2': 'Co2 Emissions',
        'ch4': 'Ch4 Emissions', 'Unnamed: 0': 'Year'})
    return fig


@emission_app.callback(
    Output('graph_histogram_ch4', 'figure'),
    Input('dropdown_histogram_ch4', 'value'))
def update_graph_histogram_ch4(country_histogram_ch4):
    # get first column of df
    fig = px.histogram(ch4_df, x=country_histogram_ch4, title='Ch4 Emissions over the years for each country', labels={
        'value': 'Ch4 Emissions', 'variable': 'Country', 'year': 'Year', 'index': 'Year', 'co2': 'Co2 Emissions',
        'ch4': 'Ch4 Emissions', 'Unnamed: 0': 'Year'})
    return fig


@emission_app.callback(
    Output('graph_correlation_co2', 'figure'),
    Input('dropdown_correlation_co2', 'value'))
def update_graph_correlation_co2(country_corr_co2):
    country_co2_data = get_country_from_data_with_index(co2_df, country_corr_co2)
    corr_df = population_and_co2(country_corr_co2, country_co2_data, population_df)
    fig = px.scatter(corr_df, x='co2', y=country_corr_co2, trendline='ols')

    fig.update_layout(
        title={
            'text': f"Correlation between CO2 and population for {corr_df.index.name}",
            'x': 0.5
        },
        xaxis_title="CO2 emissions",
        yaxis_title="Population",
        yaxis_type="log"
    )
    return fig


@emission_app.callback(
    Output('graph_correlation_ch4', 'figure'),
    Input('dropdown_correlation_ch4', 'value'))
def update_graph_correlation_ch4(country_corr_ch4):
    country_ch4_data = get_country_from_ch4_data_with_index(ch4_df, country_corr_ch4)
    corr_df = population_and_ch4(country_corr_ch4, country_ch4_data, population_df)
    fig = px.scatter(corr_df, x='ch4', y=country_corr_ch4, trendline='ols')

    fig.update_layout(
        title={
            'text': f"Correlation between CH4 and population for {corr_df.index.name}",
            'x': 0.5
        },
        xaxis_title="CH4 emissions",
        yaxis_title="Population",
        yaxis_type="log"
    )
    return fig


@emission_app.callback(
    Output('graph_clustering_co2', 'figure'),
    Input('dropdown_clustering_co2', 'value'))
def update_graph_clustering_co2(gas_type):
    if gas_type == 'Co2':
        df = pd.read_csv("clean/co2.csv", sep=",", index_col=0)

    if gas_type == 'CH4':
        df = pd.read_csv("clean/ch4.csv", sep=",", index_col=0)
    df_transposed = df.transpose()
    df_transposed.dropna(inplace=True)

    # Scale the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_transposed)

    # Perform clustering
    kmeans = KMeans(n_clusters=7, n_init=10, max_iter=300, random_state=0)
    kmeans.fit(df_scaled)

    # Reduce the dimensionality of the data
    pca = PCA(n_components=2)
    pca.fit(df_scaled)
    X_pca = pca.transform(df_scaled)

    # Plot the clusters
    country_names = df_transposed.index
    fig = go.Figure(data=[go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers',
                                     marker=dict(color=kmeans.labels_, size=10),
                                     text=country_names)])  # Add the name of the countries as text
    fig.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2"
    )
    return fig


emission_app.run_server(debug=True)
