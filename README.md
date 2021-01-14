![MOFA+ in Python](img/mofax_header.png)

Explore MOFA+ models with [mofax](https://github.com/gtca/mofax) in your browser. Powered by [Dash](https://plot.ly/dash/) from plotly.

See [more on MOFA+ here](https://biofam.github.io/MOFA2/). See [more on Dash apps here](https://dash.plot.ly/).

## Getting started

This library offers an interactive web dashboard to explore trained MOFA+ models and investigate latent factors and original features that drive variation in your data. Built on top of [mofax](https://github.com/gtca/mofax), it makes it possible to work with fairly large models, which is especially important for single-cell omics data with increasingly large numbers of cells being profiled.

### Installation

```
pip install git+https://github.com/gtca/mofax-dash
```

### Launching

Provide a trained MOFA+ model when calling `mofax` to open it in a web dashboard:

```
mofax mofa_model.hdf5
```

## Notes

To provide more efficiency when working with large datasets and large models, the data is not uploaded to the web browser (as in the [MOFA+ shiny app](https://github.com/gtca/mofaplus-shiny)), and a connection to the HDF5 file is created by the server instead. 
