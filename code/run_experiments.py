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

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_models import train
from evaluations import (eval_model, get_best_model, calculate_swe_corr,
                         calculate_scf_corr, calculate_et_corr)
from plots import (plot_model_accuracies, plot_discharge, plot_correlations,
                   plot_snow_cell, plot_scf_cell, plot_et_cell,
                   plot_inputs)
from utils import best_camels_seed, get_camels_nse


parser = argparse.ArgumentParser()
parser.add_argument('--camels_root', type=str, required=True,
                    help='Full path to CAMELS data set.')
parser.add_argument('--tensorboard', type=bool, default=False,
                    help='Write TensorBoard logs during training')
parser.add_argument('--mode', type=str, required=True,
                    help="One of ['train', 'eval', 'both']")
args = parser.parse_args()
cfg = vars(args)

# convert camels path as PosixPath object
cfg["camels_root"] = Path(cfg["camels_root"])

# list of random seeds
seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999]

# read-in basin list
with open('snow_basins.txt', 'r') as fp:
    basins = fp.readlines()
basins = [basin.strip() for basin in basins]

# Check if run includes model training
if cfg["mode"] in ['train', 'both']:
    # create directory tree to store results
    cfg["save_dir"] = Path(__file__).absolute().parent / 'results'
    if cfg["save_dir"].is_dir():
        raise Exception("Result directory already exist. Please remove first.")
    else:
        cfg["model_dir"] = cfg["save_dir"] / 'models'
        for basin in basins:
            for seed in seeds:
                run_dir = cfg["model_dir"] / basin / f"seed_{seed}"
                run_dir.mkdir(parents=True)

    # train one model at the time with each of the random seeds
    for i, basin in enumerate(basins):
        print(f"### {i+1}/{len(basins)} Start training LSTMs for basin {basin}")
        for seed in seeds:
            print(f"## Setting random seed to {seed}")
            train(basin, seed, cfg)

# check if run includes evaluation
if cfg["mode"] in ["eval", "both"]:

    # add path to data dir to cfg dictionary and create dir to store plots
    cfg["data_dir"] = Path(__file__).absolute().parent / 'data'
    cfg["plot_dir"] = Path(__file__).absolute().parent / 'plots'
    if not cfg["plot_dir"].is_dir():
        cfg["plot_dir"].mkdir()

    results = defaultdict(dict)

    # check if trained models exist
    if "model_dir" not in cfg.keys():
        cfg["save_dir"] = Path(__file__).absolute().parent / 'results'
        cfg["model_dir"] = cfg["save_dir"] / 'models'
        if not cfg["model_dir"].is_dir():
            msg = ["No result folder found. Please run this script first with ",
                   "'--mode train' or '--mode both'"]
            raise RuntimeError("".join(msg))

    # determine the best seed of the SAC-SMA models for comparison
    if not (cfg["data_dir"] / "sac_sma_seeds.txt").is_file():
        best_camels_seed(basins, cfg)

    for basin in basins:
        best_weights, best_nse = get_best_model(cfg, basin)

        results["val_NSE"][basin] = best_nse
        print(f"==> Basin {basin}: NSE of validation period: {best_nse:.4f}")

        # evaluate test set
        preds, obs, h_ns, c_ns, nse, ds_test = eval_model(basin, best_weights,
                                                          cfg, verbose=True)
        results["test_NSE"][basin] = nse

        # calculate SWE correlation
        rhos, ps, cell = calculate_swe_corr(basin, c_ns.numpy(), ds_test, cfg)
        results["SWE_rhos"][basin] = rhos
        results["SWE_ps"][basin] = ps
        results["SWE_cell"][basin] = cell

        # calculate correlation with MODIS SCF
        rho, p, cell = calculate_scf_corr(basin, c_ns.numpy(), ds_test, cfg)
        results["SCF_rho"][basin] = rho
        results["SCF_p"][basin] = p
        results["SCF_cell"][basin] = cell

        # caculate correlation with MODIS ET
        rho, p, cell = calculate_et_corr(basin, h_ns.numpy(), ds_test, cfg)
        results["ET_rho"][basin] = rho
        results["ET_p"][basin] = p
        results["ET_cell"][basin] = cell

    # calculate NSEs of SAC-SMA + Snow-17 for comparison
    results["camels"] = get_camels_nse(basins, cfg)

    print("Create boxplot of model accuracies")
    # create boxplot of validation and test accuracies
    plot_model_accuracies(results, (5, 4), cfg)

    print("Create boxplot with correlations to hyd. system states and fluxes")
    # create boxplot of basin average correlations
    plot_correlations(results, (8, 4), cfg)

    print("Create plot of SWE vs LSTM memory cell for specific basin")
    # use the following basin as special basin for inspection
    basin = '01055000'
    best_weights, best_nse = get_best_model(cfg, basin)
    preds, obs, h_ns, c_ns, nse, ds_test = eval_model(basin, best_weights,
                                                      cfg, verbose=False)
    # plot snow cell
    plot_snow_cell(basin, c_ns.numpy(), results["SWE_cell"][basin], sample=650,
                   cfg=cfg, figsize=(14, 7))

    print("Create plot of input variables")
    plot_inputs(basin, sample=0, cfg=cfg, figsize=(20, 2))

    print("Create plot of simulated vs observed discharge")
    plot_discharge(obs.numpy(), preds.numpy(), ds_test, (8, 4), cfg, basin,
                   results["test_NSE"][basin])

    print("Create plot of SCF vs LSTM memory cell of median basin")
    # only look at positive correlations here for educational reasons
    rhos = list(results["SCF_rho"].values())
    rhos = [rho for rho in rhos if rho > 0]
    scf_median = np.median(rhos)
    scf = np.inf
    basin = None
    for key, val in results["SCF_rho"].items():
        if abs(scf_median - val) < scf:
            scf = abs(scf_median - val)
            basin = key

    best_weights, best_nse = get_best_model(cfg, basin)
    preds, obs, h_ns, c_ns, nse, ds_test = eval_model(basin, best_weights,
                                                      cfg, verbose=False)

    plot_scf_cell(basin, c_ns.numpy(), results["SCF_cell"][basin],
                  rho=results["SCF_rho"][basin], p=results["SCF_p"][basin],
                  cfg=cfg, figsize=(10, 4))

    print("Create plot of ET vs LSTM cell output of median basin")
    # only look at positive correlations here for educational reasons
    rhos = list(results["ET_rho"].values())
    rhos = [rho for rho in rhos if rho > 0]
    et_median = np.median(rhos)
    et = np.inf
    basin = None
    for key, val in results["ET_rho"].items():
        if abs(et_median - val) < et:
            et = abs(et_median - val)
            basin = key

    best_weights, best_nse = get_best_model(cfg, basin)
    preds, obs, h_ns, c_ns, nse, ds_test = eval_model(basin, best_weights,
                                                      cfg, verbose=False)

    plot_et_cell(basin, h_ns.numpy(), results["ET_cell"][basin],
                 rho=results["ET_rho"][basin], p=results["ET_p"][basin],
                 cfg=cfg, figsize=(10, 4))




