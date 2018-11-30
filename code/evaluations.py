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
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from camelstxt import CamelsTXT
from model import Model
from utils import calc_nse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_model(basin: str, weight_file: PosixPath, cfg: Dict,
               verbose: bool=True) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              float, CamelsTXT]:
    """ Evaluate LSTM on the test set of a given basin.

    :param basin: 8-digit basin code of CAMELS data set
    :param weight_file: Path to the trained LSTM weights
    :param cfg: Dictionary containing several run options
    :param verbose: whether to print status outputs or now
    :return: Torch Tensors containing predictions, observations, hidden states
        and cell states for each sample in the test set, as well as the NSE
        and the CamelsTXT data set.
    """
    # create model and load pre-trained weights
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(weight_file))

    # get feature-wise mean and std of training set
    ds_train = CamelsTXT(cfg["camels_root"], basin, period="n_cal",
                         years=[0, 20])
    means = ds_train.get_means()
    stds = ds_train.get_stds()

    # prepare test data
    ds_test = CamelsTXT(cfg["camels_root"], basin, period="n_test",
                        years=[25, -1], means=means, stds=stds)
    loader = DataLoader(ds_test, batch_size=1024)

    # Variable to concatenate batch results
    preds, obs, h_ns, c_ns = None, None, None, None

    # start evaluation loop
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred, h_n, c_n = model(x)

            if preds is None:
                preds = pred.cpu()
                obs = y
                h_ns = h_n.cpu()
                c_ns = c_n.cpu()
            else:
                preds = torch.cat((preds, pred.cpu()), 0)
                obs = torch.cat((obs, y), 0)
                h_ns = torch.cat((h_ns, h_n.cpu()), 0)
                c_ns = torch.cat((c_ns, c_n.cpu()), 0)

    # rescale model predictions to original feature scale
    preds = ds_test.local_rescale(preds, variable="output")

    # clip negative discharges to zero
    preds[preds < 0] = 0

    # calculate NSE where valid observations exist
    nse = calc_nse(obs[obs >= 0].numpy(), preds[obs >= 0].numpy())

    if verbose:
        print(f"==> Basin {basin}: NSE of test period: {nse:.4f}")

    return preds, obs, h_ns, c_ns, nse, ds_test


def get_best_model(cfg: Dict, basin: str) -> Tuple[PosixPath, float]:
    """Determine the best model for each basin.

    :param cfg: Dictionary containing several run options
    :param basin: 8-digit basin code of CAMELS data set
    :return: Path to LSTM weights of best model as well as the validation NSE
        of this model.
    """
    basin_dir = cfg["model_dir"] / basin
    weight_files = list(basin_dir.glob("**/*epoch40*.pt"))

    # get seed with best accuracy at the end of training epochs
    best_weights = None
    best_nse = -np.inf
    for weight_file in weight_files:
        parts = weight_file.name.split('_')
        nse = float(parts[2][3:-3])
        if nse > best_nse:
            best_nse = nse
            best_weights = weight_file

    return best_weights, best_nse


def calculate_swe_corr(basin: str, c_ns: np.ndarray, ds: CamelsTXT,
                       cfg: Dict) -> Tuple[List, List, int]:
    """Calculate mean Spearman corr of LSTM cell and Snow-water-equivalent.

    Correlation is calculate for each sample in the test set.

    :param basin: 8-digit basin code of CAMELS data set
    :param c_ns: Array containing the cell states of each sample
    :param ds: CamelsTXT data set, which is needed to infer the date of each
        sample.
    :param cfg: Dictionary containing several run options
    :return: List of correlation coefficient for each sample, as well as p-value
        and the cell number of the cell with the highest correlation.
    """
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

    # to reduce the computational costs, we first determine the cell with the
    # highest correlation against SWE and then only calculate the correlation
    # of this cell over the entire period
    sample = 0
    start_date = ds.period_start + pd.DateOffset(days=sample)
    end_date = start_date + pd.DateOffset(days=365 - 1)

    max_rho, cell = -np.inf, None
    swe_seq = df.loc[start_date:end_date, 'SWE'].values
    for i in range(c_ns.shape[-1]):
        rho, p = spearmanr(c_ns[sample, :, i], swe_seq)
        if abs(rho) > max_rho:
            max_rho = abs(rho)
            cell = i

    # now calculate corr over entire period for this cell
    rhos = []
    ps = []
    print(f"Calculate SWE correlation for basin {basin}")
    for i in tqdm(range(c_ns.shape[0])):
        start_date = ds.period_start + pd.DateOffset(days=i)
        end_date = start_date + pd.DateOffset(days=365 - 1)

        # account for the few basins, where SAC-SMA was not simulated until end
        if end_date > df.index[-1]:
            break
        else:
            rho, p = spearmanr(c_ns[i, :, cell],
                               df.loc[start_date:end_date, 'SWE'].values)
            rhos.append(rho)
            ps.append(p)

    return rhos, ps, cell


def calculate_scf_corr(basin: str, c_ns: np.ndarray, ds: CamelsTXT,
                       cfg: Dict) -> Tuple[float, float, int]:
    """Calculate Spearman correlation of LSTM cells and snow cover fraction.

    The last cell state of each sample is concatenate as one time series
    compared against snow cover fraction derived from MODIS.

    :param basin: 8-digit basin code of CAMELS data set
    :param c_ns: Array containing the cell states of each sample
    :param ds: CamelsTXT data set, which is needed to infer the date of each
        sample.
    :param cfg: Dictionary containing several run options
    :return: Value of max (abs) correlation, p-value and cell number.
    """
    # load snow-cover fraction
    scf_file = cfg["data_dir"] / 'SCF_basins_MOD10A2.txt'
    df_scf = pd.read_csv(scf_file, sep=" ", header=0, parse_dates=True,
                         index_col="date")
    columns = df_scf.columns
    # add leading 0 to 7-digit codes
    columns = [c.zfill(8) if (len(c) == 7) else c for c in columns]
    df_scf.columns = columns

    # second data frame with last cell state of each sequence
    dates = pd.date_range(start=ds.period_start + pd.DateOffset(days=364),
                          end=ds.period_end)
    df = pd.DataFrame(data={f"cell_{i}": c_ns[:, -1, i] for i in range(10)},
                      index=dates)

    # copy snow cover fraction of basin of interest into cell state data frame
    df["snow_cover"] = df_scf[basin]

    # drop all days without scf measurement
    df = df.dropna(axis=0)

    # find cell with highest (absolute) correlation
    rhos, ps = spearmanr(df.values)
    cell = abs(rhos[:-1, -1]).argmax()
    max_rho = rhos[cell, -1].max()
    p = ps[cell, -1]
    return max_rho, p, cell


def calculate_et_corr(basin: str, h_ns: np.ndarray, ds: CamelsTXT,
                      cfg: Dict) -> Tuple[float, float, int]:
    """Calculate Spearman correlation of LSTM cell output and evapotranspiration

    The last hidden state of each sample is concatenate as one time series
    compared against evapotranspiration derived from MODIS.

    :param basin: 8-digit basin code of CAMELS data set
    :param h_ns: Array containing the hidden states of each sample
    :param ds: CamelsTXT data set, which is needed to infer the date of each
        sample.
    :param cfg: Dictionary containing several run options
    :return: Value of max (abs) correlation, p-value and cell number.
    """
    # load MODIS et values
    et_file = cfg["data_dir"] / 'ET_basins_MOD16A2.txt'
    df_et = pd.read_csv(et_file, sep=" ", header=0, parse_dates=True,
                        index_col="date")
    columns = df_et.columns
    # add leading 0 to 7-digit codes
    columns = [c.zfill(8) if (len(c) == 7) else c for c in columns]
    df_et.columns = columns

    # second data frame with last cell state of each sequence
    dates = pd.date_range(start=ds.period_start + pd.DateOffset(days=364),
                          end=ds.period_end)
    df = pd.DataFrame(data={f"unit_{i}": h_ns[:, -1, i] for i in range(10)},
                      index=dates)

    # copy et of basin of interest into cell state data frame
    df["et"] = df_et[basin]

    # drop all days without scf measurement
    df = df.dropna(axis=0)

    # find cell with highest (absolute) correlation
    rhos, ps = spearmanr(df.values)
    cell = abs(rhos[:-1, -1]).argmax()
    max_rho = rhos[cell, -1].max()
    p = ps[cell, -1]
    return max_rho, p, cell
