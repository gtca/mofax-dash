# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objs as go

from os import path
from sys import argv
import base64

import mofax as mfx


UPLOAD_DIRECTORY = ""


external_stylesheets = ['assets/default.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


card_content_1 = [
    dbc.CardHeader("Card header"),
    dbc.CardBody(
        [
            html.H5("Card title", className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]

card_content_2 = dbc.CardBody(
    [
        html.Blockquote(
            [
                html.P(
                    "A learning experience is one of those things that says, "
                    "'You know that thing you just did? Don't do that.'"
                ),
                html.Footer(
                    html.Small("Douglas Adams", className="text-muted")
                ),
            ],
            className="blockquote",
        )
    ]
)

card_content_3 = [
    dbc.CardImg(src="/assets/images/placeholder286x180.png", top=True),
    dbc.CardBody(
        [
            html.H5("Card with image", className="card-title"),
            html.P(
                "This card has an image on top, and a button below",
                className="card-text",
            ),
            dbc.Button("Click me!", color="primary"),
        ]
    ),
]


cards = dbc.CardColumns(
    [
        dbc.Card(card_content_1, color="primary", inverse=True),
        dbc.Card(card_content_2, body=True),
        dbc.Card(card_content_1, color="secondary", inverse=True),
        dbc.Card(card_content_3, color="info", inverse=True),
        dbc.Card(card_content_1, color="success", inverse=True),
        dbc.Card(card_content_1, color="warning", inverse=True),
        dbc.Card(card_content_1, color="danger", inverse=True),
        dbc.Card(card_content_3, color="light"),
        dbc.Card(card_content_1, color="dark", inverse=True),
    ]
)

card = html.Div(className="card", children=[
    html.Div(className="card-maintext", children="MAIN"),
    html.Div(className="card-subtext", children="subtext of the card")
])

card_dim_n = html.Div(id="card-dim-n", className="card card-small", children=[
    html.Div(className="card-maintext", children="MAIN"),
    html.Div(className="card-subtext", children="subtext of the card")
])


card_dim_d = html.Div(id="card-dim-d", className="card card-small", children=[
    html.Div(className="card-maintext", children="MAIN"),
    html.Div(className="card-subtext", children="subtext of the card")
])

card_factors = html.Div(id="card-factors", className="card", children=[
    
    
])

card_r2 = html.Div(id="card-r2", className="card", children=[
    
    
])

cards = html.Div(id="cardboard", children=[card_dim_n, card_dim_d, card_factors, card_r2])

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
        html.Div(id='output-data-upload'),
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

@app.callback(Output('card-dim-n', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_n(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        dim_n = model.shape[0]
        dim_g = len(model.groups)
        return make_card(dim_n, "samples", dim_g, "group")


@app.callback(Output('card-dim-d', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_d(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        dim_d = model.shape[1]
        dim_m = len(model.views)
        return make_card(dim_d, "features", dim_m, "view")


# @app.callback(Output('card-factors', 'children'),
#               [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename')])
# def update_factors(contents, filename):
#     if filename is not None:
#         model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
#         df = model.get_factors(factors=[0, 1], df=True)
#         fig = dcc.Graph(id="factors-plot", figure={
#             'data': [
#                 {
#                     'x': df["Factor1"],
#                     'y': df["Factor2"],
#                     'mode': 'markers',
#                     'marker': {'size': 10, 'opacity': .5}
#                 }
#             ],
#             'layout': {
#                 'clickmode': 'event+select',
#                 'xaxis': {
#                     'title': "Factor1",
#                 },
#                 'yaxis': {
#                     'title': "Factor2",
#                 },
#             }
#         })
#         return fig

def update_factors(contents, filename):
    if filename is not None:
        model = mfx.mofa_model(path.join(UPLOAD_DIRECTORY, filename))
        df = model.get_factors(factors=[0, 1], df=True)
        fig = dcc.Graph(id="factors-plot", figure={
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
        })
        return fig

app.callback(Output('card-factors', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])(update_factors)

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
        return html.Div(filename)

if __name__ == '__main__':
    # Parse first agument as the model filename, if any
    if len(argv) > 1:
        filename = argv[1]
        assert path.exists(filename), f"File {filename} does not exist"
        # model = mfx.mofa_model(filename)

        # TODO: refactor code so that it could be done cleaner
        # TODO: make a local execution (main.py?) and app.py separate files
        app.layout.children[1].children[0].children[0] = update_factors(None, filename)

    app.run_server(debug=True)
