################################################################################
#                                                                              #
# This code is part of the following publication:                              #
#                                                                              #
# F. Kratzert, M. Herrnegger, D. Klotz, S. Hochreiter, G. Klambauer            #
# "Do internals of neural networks make sense in the context of hydrology?"    #
# presented at 2018 Fall Meeting, AGU, Washington D.C., 10-14 Dec.             #
#                                                                              #
# Corresponding author: Frederik Kratzert (f.kratzert(at)gmail.com)            #
#                                                                              #
################################################################################

from pathlib import PosixPath
import sqlite3
from typing import Dict, List, Tuple

from numba import njit
import numpy as np
import pandas as pd

from tqdm import tqdm


def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    obs = obs
    sim = sim

    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator
    return nse_val


def calc_rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Root mean squared Error"""
    return np.sqrt(np.mean((obs - sim)**2))


def get_camels_nse(basins: List, cfg: Dict) -> Dict:
    """Calculate the NSE of the SAC-SMA + Snow-17

    :param basins:
    :param cfg:
    :return:
    """
    results = {}
    print("Calculating NSE for all basins of SAC-SMA + Snow-17 model...")
    for basin in tqdm(basins):
        # get seed of best performing SAC-SMA model
        sac_seed_file = cfg["data_dir"] / 'sac_sma_seeds.txt'
        df_seeds = pd.read_csv(sac_seed_file, sep=';', header=0,
                               dtype={'basin': str, 'seed': str})
        df_seeds = df_seeds.set_index(df_seeds.basin)
        sac_seed = df_seeds.loc[df_seeds.index == basin, 'seed'].values[0]

        # get HUC of basin from attribute db
        db_file = cfg["data_dir"] / 'attributes.db'
        with sqlite3.connect(str(db_file)) as conn:
            df_attr = pd.read_sql("SELECT * FROM 'basin_attributes'", conn,
                                  index_col='gauge_id')
        huc = df_attr.loc[df_attr.index == basin, 'huc'].values[0]

        # load SAC-SMA model output file
        mod_file = (cfg["camels_root"] / "model_output_daymet" / "model_output" /
                    "flow_timeseries" / "daymet" / huc /
                    f"{basin}_{sac_seed}_model_output.txt")
        df = pd.read_csv(mod_file, sep='\s+')
        dates = df.YR.map(str) + "/" + df.MNTH.map(str) + "/" + df.DY.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        start_cal = pd.to_datetime(f"{df.index[0].year}/10/01",
                                   format="%Y/%m/%d")
        start_val = start_cal + pd.DateOffset(years=15)
        val = df[start_val:]

        obs = val["OBS_RUN"].values
        sim = val["MOD_RUN"].values

        nse = calc_nse(obs=obs[obs >= 0], sim=sim[obs >= 0])
        results[basin] = nse

    return results


def best_camels_seed(basins: List, cfg:Dict):
    """Determine best model of each basin provided with CAMELS dataset.

    This function determines the seed of the best model model for each basin.
    By definition in the paper, the best seed and thus the model taken for
    validation is the one, that has the lowest RMSE in the calibration period.
    The results will be written into a txt file.

    Because there exist duplicate model output files in some huc folders,
    first basins and their correct huc will be determined from the met input
    folder.
    """
    # Available seeds
    seeds = ['05', '11', '27', '33', '48', '59', '66', '72', '80', '94']

    # determine basins and their corresponding huc
    input_folder = cfg["camels_root"] / 'basin_mean_forcing' / 'daymet'
    basin2huc = {f.name[:8]: f.parts[-2] for f in input_folder.glob('**/*.txt')}

    # drop all basins that are not of interest
    available_basins = list(basin2huc.keys())
    for key in available_basins:
        if key not in basins:
            basin2huc.pop(key, None)

    # get list of model output files only of those, belonging really to the huc
    output_folder = (cfg["camels_root"] / 'model_output_daymet' /
                     'model_output' / 'flow_timeseries' / 'daymet')
    mod_files = []
    for basin, huc in basin2huc.items():
        huc_folder = output_folder / huc
        files = [f for f in huc_folder.glob(f'{basin}*_model_output.txt')]
        mod_files = mod_files + files

    # create strings for column header in dataframe
    col_names = []
    for seed in seeds:
        col_names.append('mod' + seed + '_cal_rmse')

    # create empty dataframe to store results
    df_rmse = pd.DataFrame(data=None, index=basins, columns=col_names,
                           dtype=float)

    print("Start to determine best seed of SAC-SMA models.")
    # iterate over all model files and calculate metrics
    for f in tqdm(mod_files):

        # get seed number and basin of current file
        seed = f.name[-19:-17]
        basin = f.name[-28:-20]

        # read in the content of the model outputs
        df = pd.read_csv(f, sep='\s+', header=0)

        # create datestring to convert index to datetime format
        datestring = df.YR.map(str) + "/" + df.MNTH.map(str) + "/" + df.DY.map(
            str)
        df.index = pd.to_datetime(datestring, format="%Y/%m/%d")

        # determine start and end date of calibration and validation period
        start_cal = pd.to_datetime(f"{df.index[0].year}/10/01", yearfirst=True)
        end_cal = start_cal + pd.DateOffset(years=15) - pd.DateOffset(days=1)

        # remove invalid measurements (-999)
        df = df[df['OBS_RUN'] > 0]

        # subscript dataframe to calibration period for easier name handling
        cal = df[start_cal:end_cal]

        # calculate rmse
        rmse_cal = calc_rmse(cal['OBS_RUN'], cal['MOD_RUN'])

        # store results in dataframes
        df_rmse.loc[df_rmse.index == basin, f'mod{seed}_cal_rmse'] = rmse_cal

    # write column name of best performing model into new column
    df_rmse['best_col'] = df_rmse.idxmin(axis=1)

    # extract seed number to new column
    df_rmse['seed'] = df_rmse['best_col'].apply(lambda x: x[3:5])

    # sort dataframe by index
    df_rmse = df_rmse.sort_index()

    # drop all columns beside the seed number
    drop_cols = [n for n in df_rmse.columns if n != 'seed']
    df_rmse = df_rmse.drop(drop_cols, axis=1)

    out_file = cfg["data_dir"] / "sac_sma_seeds.txt"

    # write results to disk
    df_rmse.to_csv(out_file, sep=';', index_label='basin')


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row
        wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def load_forcing(camels_root: PosixPath, basin: str) -> \
        Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data of a specific basin.

    :param camels_root: Full path to the root directory of the CAMELS data set.
    :param basin: 8-digit code of basin as string.
    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin as integer.
    """
    forcing_path = camels_root / 'basin_mean_forcing' / 'daymet'
    files = list(forcing_path.glob('**/*_lump_cida_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    df = pd.read_csv(file_path, sep='\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
             + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # load area from header
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


def load_discharge(camels_root: PosixPath, basin: str, area: int) -> \
        Tuple[pd.Series, pd.DatetimeIndex]:
    """Load the discharge time series for a specific basin.

    :param camels_root: Full path to the root directory of the CAMELS data set.
    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters
    :return: A pd.Series containng the catchment normalized discharge and
        DateTimeIndex corresponding to the start of the calibration period,
        which is defined as 1 Oct in the first year of observed discharge.
    """
    discharge_path = camels_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
             + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # normalize discharge from cubic feed per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10 ** 6)

    # derive the date of the start of the calibration period
    cal_start = pd.to_datetime(f"{df.index[0].year}/10/01", yearfirst=True)

    return df.QObs, cal_start