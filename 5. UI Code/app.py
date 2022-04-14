# Note to self: this is the most up to date/correct version of the UI as of 4/13/2022
# deploy this version of the UI

# Sources from SPRING 2021 semester: 
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

# Source for graphing from hover data: https://dash.plotly.com/interactive-graphing
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

# Sources from SPRING 2022 semester: 
# https://dash.plotly.com/basic-callbacks
# https://dash.plotly.com/dash-core-components/upload
# https://dash.plotly.com/dash-core-components/store 
# https://dash.plotly.com/sharing-data-between-callbacks
# https://community.plotly.com/t/errors-from-none-inputs-with-chained-callbacks/23513 

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.express as px 

import base64
import io


COVID_FORECASTER_OVERALL_TITLE = 'COVID Forecasters Predictions for number of total COVID cases in the US'
UPLOADED_OVERALL_TITLE = 'Uploaded Predictions for number of total COVID cases in the US'
CHLOROPLETH_TITLE = 'Chloropleth map of total COVID cases'
LINE_GRAPH_TITLE = 'Line graph of total COVID cases'
LINE_GRAPH_FIGURE_PARTIAL_TITLE = 'Predicted number of total cases in '
COLORBAR_TITLE = '# total cases'

STATE_COLUMN_NAME = 'state'
DATE_COLUMN_NAME = 'date_today'
NUMBER_COLUMN_NAME = 'cumulative_cases'

COVID_FORECASTER_FILENAME = 'COVID_Forecaster_full.csv'

# Helper function that reads in the contents of the ALREADY PRESENT LOCAL input file that contains STAN's predictions
def read_csv(filename):
    return pd.read_csv(filename)

# Helper function that reads in the contents of the USER-UPLOADED input file and returns the input file data as a pandas dataframe
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    print("in parse_contents, filename and return type of df is:", filename, type(df))

    return df


def preprocessing_for_graphs(df):
    df1 = df.copy() # Make a copy of the dataframe to ensure we don't accidentally modify the original df
    df2 = df.copy() # Need a second copy of the input dataframe for the line plot graph since we cast values of "df" to strings for the us chloropleth map

    ############################# START OF PREPROCESSING FOR US MAP GRAPH ######################################################
    scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.4, '#ff4d4d'], 
        [0.6, '#ff1a1a'],[0.8, '#cc0000'],[1.0, '#4d0000']] # reds

    for col in df1.columns:
        df1[col] = df1[col].astype(str)

    data_slider = []
    for date in df1.date_today.unique():
        df1_selected_date = df1[(df1[DATE_COLUMN_NAME] == date)]
        for col in df1_selected_date:
            df1_selected_date.loc[col] = df1_selected_date[col].astype(str)
        data_one_day_1 = dict(
                    type='choropleth', # type of map-plot
                    colorscale = scl,
                    autocolorscale = True,
                    locations = df1_selected_date.state.unique(), # the column with the state
                    z = df1_selected_date[NUMBER_COLUMN_NAME].astype(float), # the variable I want to color-code
                    locationmode = 'USA-states',
                    text = df1_selected_date[DATE_COLUMN_NAME], # hover text
                    marker = dict(     # for the lines separating states
                                line = dict (
                                        color = 'rgb(255,255,255)', 
                                        width = 2) ),               
                    colorbar = dict(
                                title = COLORBAR_TITLE)
                    )
        data_slider.append(data_one_day_1)

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
    list_states = df2['state'].unique()
    optionsArray = []
    for state in list_states:
        toAdd = {'label': str(state), 'value': str(state)}
        optionsArray.append(toAdd)
    ################################################## End of preprocessing for line graph #################################
    
    print("In preprocessing for graphs, layout is: ", type(layout))
    print("In preprocessing for graphs, fig is: ", type(fig))
    print("In preprocessing for graphs, df2 is: ", type(df2))
    return fig, df2


STAN_data = read_csv(COVID_FORECASTER_FILENAME)
STAN_fig, STAN_df2 = preprocessing_for_graphs(STAN_data)

# you need to include __name__ in your Dash constructor if
# you plan to use a custom CSS or JavaScript in your Dash apps
app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    # Div for a SINGLE ROW
    html.Div([
        html.H3(COVID_FORECASTER_OVERALL_TITLE)
    ], style = {'color': 'white', 'fontFamily': 'Playfair Display', 'fontSize': '4rem', 'textAlign': 'center'}),

    # Div for a SINGLE ROW
    html.Div([
        # Div for a SINGLE COLUMN
        html.Div([
            html.Center(html.H3(CHLOROPLETH_TITLE)),
            # Start of graph stuff
            dcc.Graph(
                id='STAN_US_map_graph',
                figure=STAN_fig,
            ),
            # End of graph stuff stuff
        ], className="six columns"),

        # Div for a SINGLE COLUMN
        html.Div([
            html.Center(html.H3(LINE_GRAPH_TITLE)),
            # Start of graph stuff
            html.Div([
                dcc.Graph(id='STAN_state_graph'),
            ]),
            # End of graph stuff stuff
        ], className="six columns"),

    ], className="row"),

    # dcc.Store(id='uploaded_data'),
    dcc.Store(id='uploaded_data'),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
    ]),
    
    # html.Div(id='output-data-upload')

    html.Div([
        # Div for a SINGLE ROW
        html.Div([
            html.H3(UPLOADED_OVERALL_TITLE)
        ], style = {'color': 'white', 'fontFamily': 'Playfair Display', 'fontSize': '4rem', 'textAlign': 'center'}),

        # Div for a SINGLE ROW
        html.Div([
            # Div for a SINGLE COLUMN
            html.Div([
                html.Center(html.H3(CHLOROPLETH_TITLE)),
                # Start of graph stuff
                dcc.Graph(
                    id='UPLOADED_US_map_graph'
                ),
                # End of graph stuff stuff
            ], className="six columns"),

            # Div for a SINGLE COLUMN
            html.Div([
                html.Center(html.H3(LINE_GRAPH_TITLE)),
                # Start of graph stuff
                html.Div([
                    dcc.Graph(id='UPLOADED_state_graph')
                ]),
                # End of graph stuff stuff
            ], className="six columns"),

        ], className="row")
    ])
])


# Input is hover data from STAN's US state map
# Output is updated line graph for state being hovered over
@app.callback(
    Output(component_id='STAN_state_graph', component_property='figure'),
    Input(component_id='STAN_US_map_graph', component_property='hoverData')
)
def update_line_graph_onHover(inputValue_hoverData):
    # Sample format of inputValue:
    # {'points': [{'curveNumber': 27, 'pointNumber': 14, 'pointIndex': 14, 'location': 'KS', 'z': 193.73505, 'text': '2021-06-09'}]}
    
    curr_state = 'AL'
    if inputValue_hoverData is not None:
        curr_state = str(inputValue_hoverData['points'][0]['location'])
    
    state_data = STAN_df2[STAN_df2[STATE_COLUMN_NAME] == curr_state]

    fig_toReturn = px.line(state_data, x=DATE_COLUMN_NAME, y=NUMBER_COLUMN_NAME, title=LINE_GRAPH_FIGURE_PARTIAL_TITLE + curr_state)
    return fig_toReturn

# Input is hover data from uploaded csv's US state map
# Output is updated line graph for state being hovered over
@app.callback(
    Output(component_id='UPLOADED_state_graph', component_property='figure'),
    Input(component_id='UPLOADED_US_map_graph', component_property='hoverData'),
    Input('uploaded_data', 'data')
)
def update_line_graph_onHover_v2(inputValue_hoverData, uploaded_serialized_data):
    print("update_line_graph_onHover_v2 is called")
    print("inputValue_hoverData", inputValue_hoverData)
    print("uploaded_serialized_data", uploaded_serialized_data)

    # Sample format of inputValue:
    # {'points': [{'curveNumber': 27, 'pointNumber': 14, 'pointIndex': 14, 'location': 'KS', 'z': 193.73505, 'text': '2021-06-09'}]}
    if (inputValue_hoverData is None or uploaded_serialized_data is None):
        print("got to inside of IF in update_line_graph_onHover_v2")
        raise PreventUpdate
    else:
        print("got to inside of ELSE in update_line_graph_onHover_v2")
        curr_state = 'AL'
        if inputValue_hoverData is not None:
            curr_state = str(inputValue_hoverData['points'][0]['location'])
        
        # deserialize data before using it
        df = pd.read_json(uploaded_serialized_data, orient='split').copy()
        state_data = df[df[STATE_COLUMN_NAME] == curr_state]

        fig_toReturn = px.line(state_data, x=DATE_COLUMN_NAME, y=NUMBER_COLUMN_NAME, title=LINE_GRAPH_FIGURE_PARTIAL_TITLE + curr_state)
        return fig_toReturn


# Called whenever a new file is uploaded
# @app.callback(
#               Output('uploaded_data', 'data'),
#               Output('output-data-upload', 'children'),
#               Input('upload-data', 'contents'),
#               State('upload-data', 'filename'),
#               State('upload-data', 'last_modified'))
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]

#         # Serialize data when it is uploaded and store it in correct dcc.store component
#         UPLOADED_data_df = children[0]
#         UPLOADED_data_serialized = UPLOADED_data_df.to_json(date_format='iso', orient='split')

#         # Create the graph we need and return it later
#         UPLOADED_fig, _ = preprocessing_for_graphs(UPLOADED_data_df)

#         toReturn = html.Div([
#             # Div for a SINGLE ROW
#             html.Div([
#                 html.H3('Rate of daily new COVID-19 cases in the US')
#             ], style = {'color': 'white', 'fontFamily': 'Playfair Display', 'fontSize': '4rem', 'textAlign': 'center'}),

#             # Div for a SINGLE ROW
#             html.Div([
#                 # Div for a SINGLE COLUMN
#                 html.Div([
#                     html.Center(html.H3('Chloropleth map of daily new cases')),
#                     # Start of graph stuff
#                     dcc.Graph(
#                         id='UPLOADED_US_map_graph',
#                         figure=UPLOADED_fig
#                     ),
#                     # End of graph stuff stuff
#                 ], className="six columns"),

#                 # Div for a SINGLE COLUMN
#                 html.Div([
#                     html.Center(html.H3('Line graph of daily new cases')),
#                     # Start of graph stuff
#                     html.Div([
#                         dcc.Graph(id='UPLOADED_state_graph')
#                     ]),
#                     # End of graph stuff stuff
#                 ], className="six columns"),
#             ], className="row")
#         ])
#         return UPLOADED_data_serialized, toReturn


# Called whenever a new file is uploaded and returns the graphs to be displayed
# Input is csv file that is uploaded
# Output is 1. the uploaded data serialized as a json, and 
#           2. state graph for the uploaded data
@app.callback(
              Output('uploaded_data', 'data'),
              Output('UPLOADED_US_map_graph', 'figure'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    
    print("update_output was called")

    # if list_of_contents is not None:
    #     children = [
    #         parse_contents(c, n, d) for c, n, d in
    #         zip(list_of_contents, list_of_names, list_of_dates)]

    #     # Serialize data when it is uploaded and store it in correct dcc.store component
    #     UPLOADED_data_df = children[0]
    #     UPLOADED_data_serialized = UPLOADED_data_df.to_json(date_format='iso', orient='split')

    #     # Create the graph we need and return it later
    #     UPLOADED_fig, _ = preprocessing_for_graphs(UPLOADED_data_df)

    #     print("in update output, output 1 is: ", type(UPLOADED_data_serialized))
    #     print("in update output, output 2 is: ", type(UPLOADED_fig))
    #     return UPLOADED_data_serialized, UPLOADED_fig

    if (list_of_contents is None or list_of_names is None or list_of_dates is None):
        raise PreventUpdate
    else:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        # Serialize data when it is uploaded and store it in correct dcc.store component
        UPLOADED_data_df = children[0]
        UPLOADED_data_serialized = UPLOADED_data_df.to_json(date_format='iso', orient='split')

        # Create the graph we need and return it later
        UPLOADED_fig, _ = preprocessing_for_graphs(UPLOADED_data_df)

        print("in update output, output 1 is: ", type(UPLOADED_data_serialized))
        print("in update output, output 2 is: ", type(UPLOADED_fig))
        return UPLOADED_data_serialized, UPLOADED_fig

if __name__ == '__main__':
    app.run_server(debug=True)








