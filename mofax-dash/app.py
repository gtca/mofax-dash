# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objs as go

from os import path
from sys import argv, exit
from contextlib import ExitStack
from functools import partial
import base64

import mofax as mfx


UPLOAD_DIRECTORY = ""


external_stylesheets = ['assets/default.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



card_dim_n = html.Div(id="card-dim-n", className="card card-small", children=[
    html.Div(className="card-maintext", children="MAIN"),
    html.Div(className="card-subtext", children="subtext of the card")
])


card_dim_d = html.Div(id="card-dim-d", className="card card-small", children=[
    html.Div(className="card-maintext", children="MAIN"),
    html.Div(className="card-subtext", children="subtext of the card")
])

card_dim_k = html.Div(id="card-dim-k", className="card card-small", children=[
    html.Div(className="card-maintext", children="MAIN"),
    html.Div(className="card-subtext", children="subtext of the card")
])

card_settings = html.Div(id="card-settings", className="card", children=[
    
    
])

card_r2 = html.Div(id="card-r2", className="card", children=[
    
    
])


card_factors = html.Div(id="card-factors", className="card", children=[
    
    
])

card_factors_violin = html.Div(id="card-factors-violin", className="card", children=[
    
    
])

cards = html.Div(id="cardboard", children=[card_dim_n, card_dim_d, card_dim_k, card_settings, card_r2, card_factors, card_factors_violin])

app.layout = html.Div(id='content', children=[
    html.Div(id='header', children=[
        html.H1(id='header-title', children='Hello Dash'),

        html.Div(id='header-subtitle', children='''
            MOFA+ model exploration
        '''),
    ]),

    html.Div(id='overview', children=[
        cards
    ]),

    html.Div(children=[
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
            multiple=False
        ),
        html.Div(id='output-data-upload', style={'display': 'none'}),
    ]),

    html.Footer(children=[
        html.Div(children=[
            html.A('MOFA+', href='https://github.com/bioFAM/MOFA2', target='_blank'),
            ' | ',
            html.A('mofa×', href='https://github.com/gtca/mofax', target='_blank'),
        ]),
    ]),
])



def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))




def make_title(model_filename):
    title = model_filename.strip('.hdf5').replace('_', ' ')
    return html.H1(id='header-title', children=title),

@app.callback(Output('header-title', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_title(contents, filename):
    if filename is not None:
        return make_title(filename)


def make_card(main_value, main_desc, sub_value, sub_desc):
    return html.Div(className="card", children=[
                html.Div(className="card-maintext", children=f"{main_value} {main_desc}"),
                html.Div(className="card-subtext", children=f"in {sub_value} {sub_desc}{'s' if sub_value > 1 else ''}")
            ])


def make_card_children(main_value, main_desc, sub_value, sub_desc):
    return [
                html.Div(className="card-maintext", children=f"{main_value} {main_desc}"),
                html.Div(className="card-subtext", children=f"in {sub_value} {sub_desc}{'s' if sub_value > 1 else ''}")
           ]


# Update dimensions

# @app.callback(Output('card-dim-n', 'children'),
#               [Input('output-data-upload', 'children')])
# def update_n(div):
#     dim_n = model.shape[0]
#     dim_g = len(model.groups)
#     return make_card_children(dim_n, "samples", dim_g, "group")

@app.callback(Output('card-dim-n', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_n(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        dim_d = model.shape[1]
        dim_m = len(model.views)
        return make_card_children(dim_d, "features", dim_m, "view")


@app.callback(Output('card-dim-d', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_d(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        dim_d = model.shape[1]
        dim_m = len(model.views)
        return make_card_children(dim_d, "features", dim_m, "view")


@app.callback(Output('card-dim-k', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_k(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        dim_k = model.nfactors
        return make_card_children(dim_k, "factors", dim_k, "factors")


def update_factors(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        df = model.get_factors(factors=[0, 1], df=True)
        fig = dcc.Graph(id="factors-plot-scatter", figure={
            'data': [
                {
                    'x': df["Factor1"],
                    'y': df["Factor2"],
                    'mode': 'markers',
                    'marker': {'size': 10, 'opacity': .5}
                }
            ],
            'layout': {
                'clickmode': 'event+select',
                'xaxis': {
                    'title': "Factor1",
                },
                'yaxis': {
                    'title': "Factor2",
                },
            }
        }, style={'width': '100%'})
        return fig

app.callback(Output('card-factors', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])(update_factors)




def update_factors_violin(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        df = model.get_factors(factors=None, df=True)\
                  .rename_axis("Sample")\
                  .reset_index()\
                  .melt(var_name="Factor", value_name="Value", id_vars=["Sample"])
        fig = dcc.Graph(id="factors-plot-violin", figure={
            'data': [
                {   
                    'type': 'violin',
                    'x': df["Factor"],
                    'y': df["Value"],
                    'text': df['Sample']
                    # 'points': 'all'
                }
            ],
            'layout': {
                'clickmode': 'event+select',
                'xaxis': {
                    'title': "Factor",
                },
                'yaxis': {
                    'title': "Factor value",
                },
            }
        })
        return fig

app.callback(Output('card-factors-violin', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])(update_factors_violin)




# TODO: divide into update R2 heatmap and update selectors
# see https://github.com/balajiciet/daypart/blob/master/daypart.py
def update_r2(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        df = model.get_r2()
        fig = dcc.Graph(id="r2-plot", figure={
            'data': [
                go.Heatmap(
                    x= df["Group"],
                    y= df["Factor"],
                    z= df["R2"],
                    colorscale='Purples',
                )
            ],
            'layout': go.Layout(
                clickmode= 'event+select',
                xaxis= {
                    'title': "Group",
                },
                yaxis= {
                    'title': "Factor",
                },
            )
        })

        sel  = html.Div([
            html.Div([
                html.H5('Select data subsetting'),
                dcc.Dropdown(
                   id="ViewGroupFactorDropdown",
                   options=[{'label': i, 'value': i} for i in ["View", "Group", "Factor"]],
                   value="View",
                   ),
                ],
                style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                html.H5('Select data subset'),
                dcc.Dropdown(
                    id="ViewGroupFactorDropdownChoice",
                    options=["View1", "ViewN"],
                    value="View1",
                    ),
                ],
                style={'width': '30%', 'float': 'right', 'display': 'inline-block'}),
            ])
        return [sel, fig]


app.callback(Output('card-r2', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])(update_r2)


app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        save_file(filename, contents)
        global model
        model = mfx.mofa_model(filename)
        return html.Div(filename)

if __name__ == '__main__':
    # Parse first agument as the model filename, if any
    if len(argv) > 1:
        filename = argv[1]
        assert path.exists(filename), f"File {filename} does not exist"

        global model
        model = mfx.mofa_model(filename)

        # # Use ExitStack to defer model.close()
        # with ExitStack() as stack:
        #     stack.callback(model.close())
    else:
        print("Please provide an .hdf5 with a trained MOFA+ model")
        # exit(1)
    app.run_server(debug=True)
    
