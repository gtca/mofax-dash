# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import mofax as mfx

external_stylesheets = ['assets/default.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(id='content', children=[
    html.Div(id='header', children=[
        html.H1(id='header-title', children='Hello Dash'),

        html.Div(id='header-subtitle', children='''
            MOFA+ model exploration
        '''),
    ]),

    html.Div(id='overview', children=[
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

def make_title(model_filename):
    title = model_filename.strip('.hdf5').replace('_', ' ')
    return html.H1(id='header-title', children=title),

@app.callback(Output('header-title', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_title(contents, filename):
    if filename is not None:
        return make_title(filename)


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        return html.Div(filename)

if __name__ == '__main__':
    app.run_server(debug=True)
