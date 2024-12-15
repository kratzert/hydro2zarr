"""
Utility code for converting the CAMELS-CL dataset from its raw form into a single zarr file.

Usage (from terminal):

python camels_cl_converter.py --data-path=... --output_path=...
"""

#!/usr/bin/env python
import argparse
import pathlib
import sys

import tqdm
import numpy as np
import pandas as pd
import xarray as xr

_UNITS = {
    'precip_chirps': 'mm/day',
    'precip_cr2met': 'mm/day',
    'tmax_cr2met': 'C',
    'precip_tmpa': 'mm/day',
    'streamflow_mm': 'mm/day',
    'streamflow_m3s': 'm3/s',
    'precip_mswep': 'mm/day',
    'tmean_cr2met': 'C',
    'pet_8d_modis': 'mm/day',
    'swe': 'mm',
    'tmin_cr2met': 'C',
    'pet_hargreaves': 'mm/day'
}

_FILE_NAME = "camels_cl.zarr"


def convert_camels_cl_to_zarr(data_path: str | pathlib.Path,
                              output_path: str | pathlib.Path) -> None:
    """Converts CAMELS-CL dataset into a single zarr file.

    Args:
      data_path: Path to the the CAMELS-CL data set directory including the various txt files.
      output_path: Path to the directory, where the resulting zarr file will be stored.

    Raises:
        FileExistError: If zarr file already exist at the output_dir.    
    """
    if isinstance(data_path, str):
        data_path = pathlib.Path(data_path)
    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    if (output_path / _FILE_NAME).exists():
        raise FileExistsError(f"File already exists at {output_path / _FILE_NAME}")

    # Initial ignore list of files.
    ignore_list = ['attributes', 'catch_hierarchy']

    files = [
        f for f in list(data_path.glob('*.txt')) if not any([x in f.name for x in ignore_list])
    ]

    ds_timeseries = _process_timeseries_files(files)

    ds_attributes = _load_attributes(data_path)
    ds = xr.merge([ds_timeseries, ds_attributes])
    output_path.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(output_path / _FILE_NAME)

    print(f"Finished processing CAMELS-CL. Zarr file stored at {output_path / _FILE_NAME}")


def _process_timeseries_files(files: list[pathlib.Path]) -> xr.Dataset:
    dataarrays = []
    for file in tqdm.tqdm(files, file=sys.stdout, desc="Loading txt files into memory"):
        dataarrays.append(_process_timeseries_file(file))

    ds = xr.merge(dataarrays).astype(np.float32)
    ds = ds[sorted(ds.data_vars)]
    ds.attrs = {k: _UNITS[k] for k in sorted(ds.data_vars)}
    return ds


def _process_timeseries_file(file: pathlib.Path) -> xr.DataArray:
    df = pd.read_csv(file, sep='\t+', index_col=0, engine="python")

    df.index = pd.to_datetime(df.index.map(lambda x: x.replace('"', '')), format="%Y-%m-%d")
    df.reindex = pd.date_range(df.index.values[0], df.index.values[-1], freq="1D")
    df.index.name = "date"

    # Why would you ever want to wrap numbers in quotation marks...Reverse that.
    df = df.rename({x: x.replace('"', '') for x in df.columns}, axis=1)

    # Why would you ever want to encode missing values as string of a single space..Reverse that.
    df = df.map(lambda x: np.nan if x == '" "' else float(x.replace('"', '')))

    feature_name = file.stem.split('_CAMELScl_')[-1]
    return xr.DataArray(df, name=feature_name).rename({'dim_1': 'gauge_id'})


def _load_attributes(data_path: pathlib.Path) -> xr.Dataset:
    attributes_file = data_path / '1_CAMELScl_attributes.txt'

    df = pd.read_csv(attributes_file, sep="\t", index_col="gauge_id").transpose()
    df.index.name = "gauge_id"

    # Convert all columns, where possible, to numeric. Skip columns with string attributes.
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError as _:
            continue
    
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # Convert the two date columns to Timestamps.
    df["record_period_start"] = pd.to_datetime(df["record_period_start"])
    df["record_period_end"] = pd.to_datetime(df["record_period_end"])

    df = df[sorted(df.columns)]

    return df.to_xarray()


def _get_paths() -> tuple[pathlib.Path, pathlib.Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--output-path', type=str)

    args = vars(parser.parse_args())

    return pathlib.Path(args['data_path']), pathlib.Path(args['output_path'])


if __name__ == "__main__":
    data_path, output_path = _get_paths()
    convert_camels_cl_to_zarr(data_path, output_path)
