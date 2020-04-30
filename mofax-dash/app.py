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

import numpy as np

import mofax as mfx


import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


UPLOAD_DIRECTORY = ""


external_stylesheets = ['assets/default.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



if __name__ == '__main__':
    # Parse first agument as the model filename, if any
    if len(argv) > 1:
        filename = argv[1]
        assert path.exists(filename), f"File {filename} does not exist"
        
        model = mfx.mofa_model(filename)
        model_filename = filename

        model.metadata["stage"] = [i.split('_')[1].rstrip("0123456789") for i in model.metadata.index.values]

        # Number of factors explaining more than 1% of variance 
        # to be reused later for better default view
        signif_k = 0

        # # Use ExitStack to defer model.close()
        # with ExitStack() as stack:
        #     stack.callback(model.close())
    else:
        print("Please provide an .hdf5 with a trained MOFA+ model")
        exit(1)
    


def make_title(model_filename):
    if model_filename is not None:
        title = path.basename(model_filename).strip('.hdf5').replace('_', ' ')
        return html.H1(id='header-title', children=title)
    else:
        return ""




def make_card(main_value, main_desc, sub_value, sub_desc):
    return html.Div(className="card", children=[
                html.Div(className="card-maintext", children=f"{main_value} {main_desc}"),
                html.Div(className="card-subtext", children=f"in {sub_value} {sub_desc}{'s' if sub_value > 1 else ''}")
            ])


def make_card_children(main_value, main_desc, prep, sub_value, sub_desc, more=""):
    return [
                html.Div(className="card-maintext", children=f"{main_value} {main_desc}"),
                html.Div(className="card-subtext", children=f"{prep} {sub_value} {sub_desc}{'s' if sub_value > 1 else ''} {more}")
           ]


# Dimensions

def update_n(model):
    if model is not None:
        dim_n = model.shape[0]
        dim_g = len(model.groups)
        return make_card_children(dim_n, "cells", "in", dim_g, "group")


def update_d(model):
    if model is not None:
        dim_d = model.shape[1]
        dim_m = len(model.views)
        return make_card_children(dim_d, "features", "in", dim_m, "view")


def update_k(model):
    if model is not None:
        global signif_k
        dim_k = model.nfactors
        signif_k = np.sum(model.get_r2().groupby("Factor").agg({"R2": "max"}).R2 > 0.01)
        return make_card_children(dim_k, "factors", "incl.", signif_k, "factor", more="with R2>1%")


# Settings

def update_feature_selection_prompt(value):
    return f"with top {value} feature{'s' if value > 1 else ''} per factor"

def update_feature_selection(n_clicks, factors, n_features):
    return model.get_top_features(factors=factors, n_features=n_features)


# Factors

def update_factors(factor_x, factor_y, highlight):
    logging.debug(f"Plotting factors scatter highlighting {highlight}")
    factors = list(set([factor_x, factor_y]))
    if model is not None:
        df = model.fetch_values([factor_x, factor_y, highlight])

        logging.debug("Got the dataframe to plot")

        # Determine if it's a discrete or a continuous variable
        highlight_discrete = None
        if highlight is not None:
            highlight_discrete = (df[highlight].dtype.name == "object") or (df[highlight].dtype.name == "category")
            if not highlight_discrete:
                df.sort_values(highlight, inplace=True)

        fig = dcc.Graph(id="factors-plot-scatter", figure={
            'data': [
                go.Scattergl({
                    'x': df[factor_x],
                    'y': df[factor_y],
                    'text': df.index.values,
                    'mode': 'markers',
                    'marker': {'size': 10, 'opacity': .5}
                })
            ] if highlight is None else [
                go.Scattergl({
                    'x': df[df[highlight] == i][factor_x],
                    'y': df[df[highlight] == i][factor_y],
                    'text': df.index.values,
                    'mode': 'markers',
                    'marker': {'size': 10, 'opacity': .5},
                    'name': i
                }) for i in df[highlight].unique()
            ] if highlight_discrete else [
                go.Scattergl({
                    'x': df[factor_x],
                    'y': df[factor_y],
                    'text': df.index.values,
                    'mode': 'markers',
                    'marker': {'size': 10, 'opacity': .5, 
                               'color': df[highlight],
                               'colorbar': {'title': highlight},
                               'colorscale': 'Viridis'},
                    # 'colorbar': {
                    #     'title': highlight
                    # },
                })
            ],
            'layout': {
                'clickmode': 'event+select',
                'xaxis': {
                    'title': factor_x,
                },
                'yaxis': {
                    'title': factor_y,
                },
                'margin': {
                    'l': 40,
                    'r': 10,
                    'b': 40,
                    't': 10,
                },
                'autosize': True,
            }
        })

        logging.debug("Returning dataframe for scatterplot...")
        return fig



def update_factors_violin(factors, highlight):
    if factors is not None and (factors == -1 or -1 in factors): factors = None
    if model is not None:
        df = model.get_factors(factors=factors, df=True)\
                  .rename_axis("Sample")\
                  .reset_index()\
                  .melt(var_name="Factor", value_name="Value", id_vars=["Sample"])

        highlight_discrete = None
        if highlight is not None:
            df = df.set_index("Sample").join(model.fetch_values(highlight)).rename_axis("Sample").reset_index()
            # Determine if it's a discrete or a continuous variable
            highlight_discrete = (df[highlight].dtype.name == "object") or (df[highlight].dtype.name == "category")

        fig = dcc.Graph(id="factors-plot-violin", figure={
            'data': [
                {   
                    'type': 'violin',
                    'x': df["Factor"],
                    'y': df["Value"],
                    'text': df['Sample'],
                    'points': False,
                }
            ] if highlight is None else [
                {
                    'type': 'violin',
                    'x': df[df[highlight] == i]["Factor"],
                    'y': df[df[highlight] == i]["Value"],
                    'text': df[df[highlight] == i][highlight],
                    'points': False,
                    'name': i,
                } for i in df[highlight].unique()
            ] if highlight_discrete else [
                {
                    'type': 'violin',
                    'x': df["Factor"],
                    'y': df["Value"],
                    'text': df['Sample'],
                    'points': False,
                    'marker': {'size': 10, 'opacity': .5, 
                               'color': df[highlight],
                               'colorbar': {},
                               'colorscale': 'Viridis'},
                    'colorbar': {
                        'title': highlight
                    },
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
                'margin': {
                    'l': 100,
                    'r': 10,
                    'b': 50,
                    't': 10,
                }
            }
        }, style={
            'width': '100%',
            'height': '100%',
            'margin': 0,
            'padding': 0,
        })
        return fig


def update_r2(groups, views, factors):
    # Map -1 to None
    # This is due to inability to pass None as value
    if views is not None and (views == -1 or -1 in views): views = None
    if groups is not None and (groups == -1 or -1 in groups): groups = None
    if factors is not None and (factors == -1 or -1 in factors): factors = None
    if factors is not None:
        logging.debug(f"Plotting r2 for {len(factors)} factors...")
    if model is not None:
        df = model.get_r2(groups=groups, views=views, factors=factors)
        logging.debug("Got the data frame")
        fig = dcc.Graph(id="r2-plot", figure={
            'data': [
                go.Heatmap(
                    x=df["Group"],
                    y=df["Factor"],
                    z=df["R2"],
                    colorscale='Purples',
                    colorbar={'title': "R2"},
                )
            ],
            'layout': {
                'clickmode': 'event+select',
                'xaxis': {
                    'title': "Group",
                },
                'yaxis': {
                    'title': "",
                },
                'margin': {
                    'l': 100,
                    'r': 10,
                    'b': 50,
                    't': 10,
                }
            }
        }, style={
            'width': '100%',
            'height': '100%',
            'margin': 0,
            'padding': 0,
        })

        sel = html.Div([
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
        logging.debug("Returning R2")
        return [fig]

# Weights

def update_weights(views, factors, features, highlight):
    # Map -1 to None
    # This is due to inability to pass None as value
    if views is not None and (views == -1 or -1 in views): views = None
    if factors is not None and (factors == -1 or -1 in factors): factors = None
    if factors is not None:
        logging.debug(f"Plotting weights heatmap for {len(factors)} factors...")

    if model is not None:
        w = (
            model.get_weights(views=views, factors=factors, df=True, absolute_values=False)
            .rename_axis("Feature")
            .reset_index()
        )
        # w = w[w.Feature.isin(features)]
        wm = w.melt(id_vars="Feature", var_name="Factor", value_name="Weight")
        wm["AbsWeight"] = abs(wm.Weight)
        wm["Rank"] = wm.groupby("Factor")["Weight"].rank(ascending=False)

        # If highlight is set, check if it's a feature
        if highlight is not None:
            highlight_feature = highlight in wm.Feature

        logging.debug("Got the data frame for weights heatmap")

        fig = dcc.Graph(id="weights-heatmap", figure={
            'data': [
                go.Scattergl(
                    x=wm[wm.Factor == k]["Rank"],
                    y=wm[wm.Factor == k]["Weight"],
                    mode='markers',
                    text=wm[wm.Factor == k]['Feature'],
                    marker={'size': 8, 'opacity': .5, 
                            'color': "#999999"},
                    name=k,
                ) for k in wm.Factor.unique()
            ] + [
                go.Scattergl(
                    x=wm[(wm.Factor == k) & (wm.Feature.isin(features))]["Rank"],
                    y=wm[(wm.Factor == k) & (wm.Feature.isin(features))]["Weight"],
                    mode='markers',
                    text=wm[(wm.Factor == k) & (wm.Feature.isin(features).values)]["Feature"],
                    marker={'size': 8, 'opacity': 1., 'color': "#555555"},
                    name=f"{k} highlight",
                ) for k in wm.Factor.unique()
            ] + [
                go.Scattergl(
                    x=wm[wm.Factor == k][wm.Feature == highlight]["Rank"],
                    y=wm[wm.Factor == k][wm.Feature == highlight]["Weight"],
                    mode='markers',
                    text=wm[wm.Factor == k][wm.Feature == highlight]["Feature"],
                    marker={'size': 16, 'opacity': 1., 'color': "#000000"},
                    name=highlight,
            ) for k in wm.Factor.unique()
            ],
            'layout': {
                'clickmode': 'event+select',
                'xaxis': {
                    'title': ""
                },
                'yaxis': {
                    'title': "Weight",
                },
                'margin': {
                    'l': 50,
                    'r': 20,
                    'b': 20,
                    't': 20,
                }
            }
        }, style={
            'width': '100%',
            'height': '100%',
            'margin': 0,
            'padding': 0,
        })

        return [fig]



def update_weights_heatmap(views, factors, features):
    # Map -1 to None
    # This is due to inability to pass None as value
    if views is not None and (views == -1 or -1 in views): views = None
    if factors is not None and (factors == -1 or -1 in factors): factors = None
    if factors is not None:
        logging.debug(f"Plotting weights heatmap for {len(factors)} factors...")
    if model is not None:
        w = (
            model.get_weights(views=views, factors=factors, df=True, absolute_values=True)
            .rename_axis("Feature")
            .reset_index()
        )
        w = w[w.Feature.isin(features)]
        wm = w.melt(id_vars="Feature", var_name="Factor", value_name="Weight")

        logging.debug("Got the data frame for weights heatmap")

        fig = dcc.Graph(id="weights-heatmap", figure={
            'data': [
                go.Heatmap(
                    x=wm["Feature"],
                    y=wm["Factor"],
                    z=wm["Weight"],
                    colorscale='Gray',
                    reversescale=True,
                    colorbar={'title': "Absolute<br>Weight"},
                )
            ],
            'layout': {
                'clickmode': 'event+select',
                'xaxis': {
                    'title': "",
                    'tickangle': -90,
                },
                'yaxis': {
                    'title': "",
                },
                'margin': {
                    'l': 100,
                    'r': 10,
                    'b': 100,
                    't': 10,
                },
                'tickangle': 90,
            }
        }, style={
            'width': '100%',
            'height': '100%',
            'margin': 0,
            'padding': 0,
        })

        return [fig]


def update_data_heatmap(views, groups, features):
    # Map -1 to None
    # This is due to inability to pass None as value
    if views is not None and (views == -1 or -1 in views): views = None
    if groups is not None and (groups == -1 or -1 in groups): groups = None
    if features is not None and (features == -1 or -1 in features): features = None
    
    if model is not None:
        if features is None or len(features) == 0:
            features = model.get_top_features(n_features=3)
        df = model.get_data(groups=groups, features=features, df=True).rename_axis('Sample').reset_index()
        print(df.shape)
        df = df.melt(id_vars="Sample", var_name="Feature", value_name="Value")

        logging.debug("Got the data frame for data heatmap")

        fig = dcc.Graph(id="data-heatmap", figure={
            'data': [
                go.Heatmap(
                    x=df["Sample"],
                    y=df["Feature"],
                    z=df["Value"],
                    colorscale='Viridis',
                    reversescale=False,
                    colorbar={'title': "Value"},
                )
            ],
            'layout': {
                'clickmode': 'event+select',
                'xaxis': {
                    'title': "",
                    'showticklabels': False,
                    'ticks': "",
                },
                'yaxis': {
                    'title': "",
                },
                'margin': {
                    'l': 100,
                    'r': 10,
                    'b': 20,
                    't': 10,
                }
            }
        }, style={
            'width': '100%',
            'height': '100%',
            'margin': 0,
            'padding': 0,
        })

        return [fig]



card_dim_n = html.Div(id="card-dim-n", className="card card-small", children=update_n(model))


card_dim_d = html.Div(id="card-dim-d", className="card card-small", children=update_d(model))

card_dim_k = html.Div(id="card-dim-k", className="card card-small", children=update_k(model))

card_settings = html.Div(id="card-settings", className="card", children=[
    html.Div(className="settings-row", children=[ 
        html.Div(children=[
            html.Span('Factors'),
            dcc.Dropdown(
                id="factors-selection",
                options=[{'label': "All", 'value': -1}] + [{'label': k, 'value': k} for k in [f"Factor{i+1}" for i in range(model.nfactors)]],
                value=-1 if signif_k == 0 else [f"Factor{i+1}" for i in range(signif_k)],
                multi=True,
                ),
            ],
        ),
    ]),

    html.Div(className="settings-row", children=[ 
        html.Div(children=[
            html.Span('Features contibution'),
            dcc.Dropdown(
                id="features-selection",
                options=[{'label': "All", 'value': -1}] + [{'label': i, 'value': i} for v in model.features.values() for i in v],
                value=model.get_top_features(n_features=3),
                multi=True,
                ),
            html.Button('Update', id='feature-selection-prompt-button'),
            html.Div(id="feature-selection-prompt-information"),
            dcc.Slider(
                id="feature-selection-prompt-slider",
                min=1,
                max=100,
                value=3,
                marks={
                    1: {'label': '1'},
                    5: {'label': '5'},
                    10: {'label': '10'},
                    20: {'label': '20'},
                    50: {'label': '50'},
                    100: {'label': '100'},
                }
            )],
        ),
    ]),

    html.Div(id="settings-subset", className="settings-row", children=[
        html.Div(children=[
            html.Span('Groups'),
            dcc.Dropdown(
                id="groups-selection",
                options=[{'label': "All", 'value': -1}] + [{'label': g, 'value': g} for g in model.groups],
                value=-1,
                multi=True,
                ),
            ],
        ),

        html.Div(children=[
            html.Span('Views'),
            dcc.Dropdown(
                id="views-selection",
                options=[{'label': "All", 'value': -1}] + [{'label': m, 'value': m} for m in model.views],
                value=-1,
                multi=True,
                ),
            ],
        ),
    ]),

    html.Div(className="settings-row", children=[
        html.Div(children=[
            html.Span('Cells property'),
            dcc.Dropdown(
                id="cells-highlight-selection",
                options=[{'label': k, 'value': k} for k in model.metadata.columns.values] + 
                        [{'label': i, 'value': i} for v in model.features.values() for i in v] + 
                        [{'label': i, 'value': i} for i in [f"Factor{i+1}" for i in range(model.nfactors)]],
                value=None,
                multi=False,
                ),
            ],
        ),

        html.Div(children=[
            html.Span('Features property'),
            dcc.Dropdown(
                id="features-highlight-selection",
                options=[{'label': k, 'value': k} for k in model.features_metadata.columns.values],
                value=None,
                multi=False,
                ),
            ],
        ),
    ]),
])

# card_r2 = html.Div(id="card-r2", className="card", children=update_r2(None, None, None))
card_r2 = html.Div(id="card-r2", className="card", children=[])

card_factors = html.Div(id="card-factors", className="card", children=[
    # html.Div(id="plot-factors-scatter", children=[update_factors("Factor1", "Factor2", None)]),
    html.Div(id="plot-factors-scatter", children=[]),
    html.Div(id="card-factors-selectors", children = [
        dcc.Dropdown(
          id="factors-scatter-x",
          options=[{'label': k, 'value': k} for k in [f"Factor{i+1}" for i in range(model.nfactors)]],
          value="Factor1",
          multi=False,
        ),
        dcc.Dropdown(
              id="factors-scatter-y",
              options=[{'label': k, 'value': k} for k in [f"Factor{i+1}" for i in range(model.nfactors)]],
              value="Factor2",
              multi=False,
        )
        ])
    ])

# card_factors_violin = html.Div(id="card-factors-violin", className="card", children=[update_factors_violin(None, None)])
card_factors_violin = html.Div(id="card-factors-violin", className="card", children=[])


# Weights
card_weights = html.Div(id='card-weights', className="card", children=[])
card_weights_heatmap = html.Div(id='card-weights-heatmap', className="card", children=[])

# Data
card_data_heatmap = html.Div(id='card-data-heatmap', className="card", children=[])


cards = html.Div(id="cardboard", children=[card_dim_n, card_dim_d, card_dim_k, 
                                           card_settings, card_r2, 
                                           card_factors, card_factors_violin,
                                           card_weights, card_weights_heatmap,
                                           card_data_heatmap])

app.layout = html.Div(id='content', children=[
    html.Div(id='header', children=[
        make_title(model_filename),

        html.Div(id='header-subtitle', children='''
            MOFA+ model exploration
        '''),
    ]),

    html.Div(id='overview', children=[
        cards
    ]),

    # html.Div(children=[
        # dcc.Upload(
        #     id='upload-data',
        #     children=html.Div([
        #         'Drag and Drop or ',
        #         html.A('Select Files')
        #     ]),
        #     style={
        #         'width': '100%',
        #         'height': '60px',
        #         'lineHeight': '60px',
        #         'borderWidth': '1px',
        #         'borderStyle': 'dashed',
        #         'borderRadius': '5px',
        #         'textAlign': 'center',
        #         'margin': '10px'
        #     },
        #     multiple=False
        # ),
        # html.Div(id='output-data-upload', style={'display': 'none'}),
    # ]),

    html.Footer(children=[
        html.Div(children=[
            html.A('MOFA+', href='https://github.com/bioFAM/MOFA2', target='_blank'),
            ' | ',
            html.A('mofa×', href='https://github.com/gtca/mofax', target='_blank'),
        ],
        style={
            'margin-top': '50px'
        }),
    ]),
])

# def save_file(name, content):
#     """Decode and store a file uploaded with Plotly Dash."""
#     data = content.encode("utf8").split(b";base64,")[1]
#     with open(path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
#         fp.write(base64.decodebytes(data))




# Callbacks

# Settings

app.callback(Output('feature-selection-prompt-information', 'children'),
    [Input('feature-selection-prompt-slider', 'value')])(update_feature_selection_prompt)

app.callback(Output('features-selection', 'value'),
    [Input('feature-selection-prompt-button', 'n_clicks')],
    [State('factors-selection', 'value'),
     State('feature-selection-prompt-slider', 'value')])(update_feature_selection)


# R2

app.callback(Output('card-r2', 'children'),
    [Input('groups-selection', 'value'),
     Input('views-selection', 'value'),
     Input('factors-selection', 'value')])(update_r2)

# Factors

app.callback(Output('card-factors-violin', 'children'),
    [Input('factors-selection', 'value'),
     Input('cells-highlight-selection', 'value')])(update_factors_violin)


app.callback(Output('plot-factors-scatter', 'children'),
    [Input('factors-scatter-x', 'value'),
     Input('factors-scatter-y', 'value'),
     Input('cells-highlight-selection', 'value')])(update_factors)


# Weights

app.callback(Output('card-weights', 'children'),
    [Input('views-selection', 'value'),
     Input('factors-selection', 'value'),
     Input('features-selection', 'value'),
     Input('cells-highlight-selection', 'value')])(update_weights)

app.callback(Output('card-weights-heatmap', 'children'),
    [Input('views-selection', 'value'),
     Input('factors-selection', 'value'),
     Input('features-selection', 'value')])(update_weights_heatmap)


# Data

app.callback(Output('card-data-heatmap', 'children'),
    [Input('views-selection', 'value'),
     Input('groups-selection', 'value'),
     Input('features-selection', 'value')])(update_data_heatmap)





if __name__ == "__main__":
    app.run_server(debug=True)