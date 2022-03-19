# Sources for putting 2 graphs side by side: 
# https://community.plotly.com/t/two-graphs-side-by-side/5312/2
# Using custom css scripts: https://dash.plotly.com/external-resources
#       "Just create a folder named assets in the root of your app directory and include your CSS 
#       and JavaScript files in that folder. Dash will automatically serve all of the files that are 
#       included in this folder. By default the url to request the assets will be /assets"
# Link to source of codepen_io.css stylesheet used: https://codepen.io/chriddyp/pen/bWLwgP.css

# Sources for creation of line plot with selection bar:
# https://plotly.com/python/line-charts/#what-about-dash
# https://github.com/Coding-with-Adam/Dash-by-Plotly/blob/master/Dash_Interactive_Graphs/pie.py
# https://www.youtube.com/watch?v=UYH_dNSX1DM <-- youtube "Dropdown Selector- Python Dash Plotly"
# https://www.youtube.com/watch?v=iV51JqP6y_Q <-- youtube "Pie Chart (Dropdowns) - Python Dash Plotly"

# Sources for creation of chloropleth map:
# https://amaral.northwestern.edu/blog/step-step-how-plot-map-slider-represent-time-evolu


import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# data_source = "STAN-Predictions.csv"
data_source = "NYTimes-GCN-Predictions.csv"

df = pd.read_csv(data_source)
# Need a second df for the line plot graph since we cast values of "df" to strings for the us chloropleth map
df_v2 = pd.read_csv(data_source)


############################# START OF PREPROCESSING FOR US MAP GRAPH ######################################################
year = 2021

scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.4, '#ff4d4d'], 
       [0.6, '#ff1a1a'],[0.8, '#cc0000'],[1.0, '#4d0000']] # reds

for col in df.columns:
    df[col] = df[col].astype(str)

data_slider = []
for date in df.date_today.unique():
    df_selected_date = df[(df['date_today'] == date)]
    for col in df_selected_date:
        df_selected_date.loc[col] = df_selected_date[col].astype(str)
        # df_selected_date[col] = df_selected_date[col].astype(str)
    data_one_day = dict(
                type='choropleth', # type of map-plot
                colorscale = scl,
                autocolorscale = True,
                locations = df_selected_date.state.unique(), # the column with the state
                z = df_selected_date['new_cases'].astype(float), # the variable I want to color-code
                locationmode = 'USA-states',
                text = df_selected_date['date_today'], # hover text
                marker = dict(     # for the lines separating states
                            line = dict (
                                      color = 'rgb(255,255,255)', 
                                      width = 2) ),               
                colorbar = dict(
                            title = "# new cases")
                )
    data_slider.append(data_one_day)


steps = []
for i in range(len(data_slider)):
    step = dict(method = 'restyle', 
                args = ['visible', [False] * len(data_slider)],
                label = 'Days in future {}'.format(i))
    step['args'][1][i] = True
    steps.append(step)
sliders = [dict(active = 0, pad = {"t": 1}, steps = steps)]


layout = dict(geo=dict(scope='usa',
                       projection={'type': 'albers usa'}),
              sliders=sliders)

fig = dict(data=data_slider, layout=layout) 
################################### END OF PREPROCESSING FOR US MAP GRAPH ################################################################

################################################ Start of preprocessing for line graph ####################################################
list_states = df_v2['state'].unique()

optionsArray = []
for state in list_states:
    toAdd = {'label': str(state), 'value': str(state)}
    optionsArray.append(toAdd)

################################################## End of preprocessing for line graph #################################

# you need to include __name__ in your Dash constructor if
# you plan to use a custom CSS or JavaScript in your Dash apps
app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    # Div for a SIGNLE ROW
    html.Div([
        html.H3('Rate of daily new COVID-19 cases in the US')
    ], style = {'color': 'white', 'fontFamily': 'Playfair Display', 'fontSize': '4rem', 'textAlign': 'center'}),

    # Div for a SINGLE ROW
    html.Div([
        # Div for a SINGLE COLUMN
        html.Div([
            html.Center(html.H3('Chloropleth map of daily new cases')),
            # Start of graph stuff
            dcc.Graph(
                id='us_map_graph',
                figure=fig,
            ),
            # End of graph stuff stuff
        ], className="six columns"),

        # Div for a SINGLE COLUMN
        html.Div([
            html.Center(html.H3('Line graph of daily new cases')),
            # Start of graph stuff
            # html.Div([
            #     html.Label(['Select state:']),
            #     dcc.Dropdown(
            #         id='my_dropdown',
            #         options=optionsArray,
            #         value=str(list_states[0]),
            #         multi=False,
            #         clearable=False,
            #         # style={"width": "50%"}
            #     ),
            # ]),

            html.Div([
                dcc.Graph(id='the_graph')
            ]),
            # End of graph stuff stuff
        ], className="six columns"),

    ], className="row")
])



# #---------------------------------------------------------------
# Can't have 2 callbacks with the same output <-- need to figure out how to encode allow the dropdown to 
#   also update the graph/dropdown
# TODO: Look into https://dash.plotly.com/advanced-callbacks
# @app.callback(
#     Output(component_id='the_graph', component_property='figure'),
#     [Input(component_id='my_dropdown', component_property='value')]
# )
# def update_graph(dropdown_selected_state):
#     state_data = df_v2[df_v2['state'] == dropdown_selected_state]
#     fig_2 = px.line(state_data, x="date_today", y="new_cases", title='Predicted number of new daily cases in ' + dropdown_selected_state)
#     return fig_2


# Source for graphing from hover data: https://dash.plotly.com/interactive-graphing
@app.callback(
    Output(component_id='the_graph', component_property='figure'),
    [Input(component_id='us_map_graph', component_property='hoverData')]
)
def update_line_graph_onHover(inputValue):
    # Sample format of inputValue:
    # {'points': [{'curveNumber': 27, 'pointNumber': 14, 'pointIndex': 14, 'location': 'KS', 'z': 193.73505, 'text': '2021-06-09'}]}
    curr_state = 'AL'
    if inputValue is not None:
        curr_state = str(inputValue['points'][0]['location'])
    state_data = df_v2[df_v2['state'] == curr_state]
    fig_2 = px.line(state_data, x="date_today", y="new_cases", title='Predicted number of new daily cases in ' + curr_state)
    return fig_2


if __name__ == '__main__':
    app.run_server(debug=True)