from pathlib import Path

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

_CAMELS_BR_TIMESERIES_SUBDIRS = [
    '03_CAMELS_BR_streamflow_mm_selected_catchments', '04_CAMELS_BR_streamflow_simulated',
    '05_CAMELS_BR_precipitation_chirps', '06_CAMELS_BR_precipitation_mswep', '07_CAMELS_BR_precipitation_cpc',
    '08_CAMELS_BR_evapotransp_gleam', '09_CAMELS_BR_evapotransp_mgb', '10_CAMELS_BR_potential_evapotransp_gleam',
    '11_CAMELS_BR_temperature_min_cpc', '12_CAMELS_BR_temperature_mean_cpc', '13_CAMELS_BR_temperature_max_cpc'
]

_CAMELS_BR_FEATURE_COLS = [
    'evapotransp_gleam',
    'evapotransp_mgb',
    'potential_evapotransp_gleam',
    'precipitation_chirps',
    'precipitation_cpc',
    'precipitation_mswep',
    'simulated_streamflow',
    'streamflow',
    'temperature_max'
    'temperature_mean',
    'temperature_min',
]

_CAMELS_BR_METADATA = {
    'evapotransp_gleam': 'mm/day',
    'evapotransp_mgb': 'mm/day',
    'potential_evapotransp_gleam': 'mm/day',
    'precipitation_chirps': 'mm/day',
    'precipitation_cpc': 'mm/day',
    'precipitation_mswep': 'mm/day',
    'simulated_streamflow': 'm3/s',
    'streamflow': 'mm/day',
    'temperature_max': 'C',
    'temperature_mean': 'C',
    'temperature_min': 'C',
}

_FILE_NAME = "camels_br.zarr"


def load_camels_br_attributes(data_dir: Path, basins: list[str] = []) -> pd.DataFrame:
    """Load CAMELS-BR attributes.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-BR directory. Assumes that the subdirectory 01_CAMELS_BR_attributes is located in the
        data directory root folder.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    attributes_path = Path(data_dir) / '01_CAMELS_BR_attributes'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_br_*.txt')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=' ', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')
        float_cols = df_temp.select_dtypes(include=['float64']).columns
        df_temp[float_cols] = df_temp[float_cols].astype('float32')
        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


def preprocess_camels_br_dataset(data_dir: Path, output_dir: Path):
    """Preprocess CAMELS-BR data set and create per-basin files for more flexible and faster data loading.
    
    This function will read-in all time series text files and create per-basin csv files containing all timeseries 
    features at once in a new subfolder called "preprocessed". Will only consider the 897 basin for which streamflow and
    forcings exist. Note that simulated streamflow only exists for 593 out of 897 basins.
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-BR data set containing the different subdirectories that can be downloaded as individual zip
        archives.

    Raises
    ------
    FileExistsError
        If a sub-folder called 'preprocessed' already exists in `data_dir`.
    FileNotFoundError
        If any of the subdirectories of CAMELS-BR is not found in `data_dir`, specifically the folders starting with 
        `03_*` up to `13_*`.
    """
    basin_list = []

    # Streamflow and forcing data are stored in different subdirectories that start with a numeric value each. The first
    # one is streamflow mm/d starting with 03 and the last is max temp starting with 13.
    timeseries_folders = [data_dir / subdirectory for subdirectory in _CAMELS_BR_TIMESERIES_SUBDIRS]
    if any([not p.is_dir() for p in timeseries_folders]):
        missing_subdirectories = [p.name for p in timeseries_folders if not p.is_dir()]
        raise FileNotFoundError(
            f"The following directories were expected in {data_dir} but do not exist: {missing_subdirectories}")

    # Since files is sorted, we can pick the first one, streamflow, and extract the basins names from there
    basins = [x.stem.split('_')[0] for x in timeseries_folders[0].glob('*.txt')]
    print(f"Found {len(basins)} basin files under {timeseries_folders[0].name}")

    for basin in tqdm(basins, desc="Combining timeseries data from different subdirectories into one file per basin"):
        data = {}
        for timeseries_folder in timeseries_folders:
            basin_file = list(timeseries_folder.glob(f'{basin}_*'))
            if basin_file:
                df = pd.read_csv(basin_file[0], sep=' ')
                df["date"] = pd.to_datetime(df.year.map(str) + "/" + df.month.map(str) + "/" + df.day.map(str),
                                            format="%Y/%m/%d")
                df = df.set_index('date')
                feat_col = [c for c in df.columns if c not in ['year', 'month', 'day']][0]
                data[feat_col] = df[feat_col]
        df = pd.DataFrame(data)
        old_names = list(df.columns)
        rename_dict = dict(zip(old_names, _CAMELS_BR_FEATURE_COLS))
        df = df.rename(columns=rename_dict)
        basin_list.append(df.to_xarray().astype('float32').assign_coords({"gauge_id": basin
                                                                         })[sorted(_CAMELS_BR_FEATURE_COLS)])

    attr_ds = load_camels_br_attributes(data_dir, basins).to_xarray()
    ds = xr.concat(basin_list, dim='gauge_id')
    xr.merge([ds, attr_ds])

    ds.attrs.update(_CAMELS_BR_METADATA)
    ds.chunk('auto').to_zarr(output_dir / _FILE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code to turn camels-br into a zarr store')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='Path to the unzipped CAMELS-BR')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to the output zarr store')
    args = parser.parse_args()
    data_dir = Path(args.data_path)
    output_dir = Path(args.output_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_dir}")
    preprocess_camels_br_dataset(data_dir, output_dir)