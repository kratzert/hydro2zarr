# Hydro to Zarr

This repository was started at a post-AGU hackathon, where [Tadd](https://github.com/taddyb) and 
[myself](https://github.com/kratzert) talked about all the different file formats and file layouts that are used across different
large-sample hydrology datasets. As big fans of [xarray](https://github.com/pydata/xarray) and 
[zarr](https://github.com/zarr-developers/zarr-python), we decided to create scripts that convert
existing large-sample hydrology datasets into a single zarr file per dataset.

# Current status
This repository is work in progress and for the moment we just try to write different converter 
scripts in parallel. The code is temporary and will change and be harmonized at some point.


# Setup

Create a Python environment, e.g. with Conda and make sure to install the following dependencies.

```bash
conda install -c conda-forge xarray dask netCDF4 bottleneck zarr
```

# Usage

From the terminal, run the different Python script with the required input arguments. In most cases 
this should be `--data-path=...` and `--output-path=...`. But since things are in early stages, make
sure to take a look at the code.