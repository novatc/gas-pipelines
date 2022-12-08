from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

app = Dash(__name__)

# get iris data from plotly
df = pd.read_csv('clean/co2.csv')

app.layout = html.Div(children=[
    html.H1(children='Hello DoorDash'),
    dcc.Dropdown(
        id='dropdown',
        options=df.columns,
        value=['Germany', 'United States', 'China'],
        multi=True
    ),
    dcc.Graph(id='graph')

])


@app.callback(
    Output('graph', 'figure'),
    Input('dropdown', 'value'))
def update_graph(dim):
    # get first column of df
    fig = px.scatter(df,x=df.columns[0], y=dim, )
    return fig


app.run_server(debug=True)
