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

import sqlite3
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from camelstxt import CamelsTXT

np.random.seed(237)


def plot_model_accuracies(results: Dict, figsize: Tuple[float, float],
                          cfg: Dict):
    data = [list(results["val_NSE"].values()),
            list(results["test_NSE"].values()),
            list(results["camels"].values())]

    x_labels = ['Validation set', 'Test set', 'SAC-SMA + Snow-17\nTest(*)']
    y_label = 'Nash-Sutcliffe efficiency'
    title = 'Model accuracies'

    file_name = cfg["plot_dir"] / 'boxplot_model_accuracies.svg'
    generate_boxplot(data, x_labels, y_label, title, figsize, file_name)


def plot_correlations(results: Dict, figsize: Tuple[float, float], cfg: Dict):
    # calculate the correlation for all samples with a p-value < 1e-5
    swe = []
    for key, val in results["SWE_rhos"].items():
        ps = np.array(results["SWE_ps"][key])
        rhos = np.array([abs(v) for v in val])
        swe.append(np.median(rhos[ps < 1e-5]))

    scf = []
    for key, val in results["SCF_rho"].items():
        if results["SCF_p"][key] < 1e-5:
            scf.append(abs(val))

    et = []
    for key, val in results["ET_rho"].items():
        if results["ET_p"][key] < 1e-5:
            et.append(abs(val))

    file_name = cfg["plot_dir"] / 'boxplot_correlations.svg'
    x_labels = ["Snow Water Equivalent", "Snow Cover Fraction",
               "Evapotranspiration"]
    title = "Correlation of cell states with hydrological system states/fluxes"
    generate_boxplot([swe, scf, et], x_labels, 'Correlation [-]', title,
                     figsize, file_name)


def generate_boxplot(data: List, x_labels: List, y_label: str, title: str,
                     figsize: Tuple[float, float], file_name: str):
    fig, ax = plt.subplots(figsize=figsize)

    boxprops = {'linewidth': 0.5}
    medianprops = {'linewidth': 0.5}
    whiskerprops = {'linewidth': 0.5}
    meanprops = {'marker': 'D', 'markersize': 3,'markeredgewidth': 0.3,
                 'markeredgecolor': 'black'}
    flierprops = {'marker': '+', 'markersize': 3,
                  'markeredgewidth': 0.5, 'alpha': 0.3, 'color': 'black'}
    bps = ax.boxplot(data, boxprops=boxprops,
                     medianprops=medianprops, flierprops=flierprops,
                     whiskerprops=whiskerprops, capprops=whiskerprops,
                     notch=False, showmeans=True, meanprops=meanprops,
                     whis=[5, 95], patch_artist=True)


    colors = [('#a6cee3', '#1f78b4'), ('#b2df8a', '#33a02c'),
              ('#fb9a99', '#e31a1c')]

    colors = [colors[i] for i in range(len(data))]

    for prop in ['boxes', 'means', 'medians']:
        for elem, color in zip(bps[prop], colors):
            if prop == 'boxes':
                elem.set(facecolor=color[0], alpha=0.2, color=color[1])
            if prop == 'medians':
                elem.set(color=color[1])
            if prop == 'means':
                elem.set(markerfacecolor=color[0])

    # add acatter points to boxplots
    for i, d in enumerate(data):
        x = np.random.normal(i+1-0.3, 0.04, len(d))
        ax.scatter(x, d, alpha=0.6, s=6, c=colors[i][1], edgecolor=colors[i][1],
                   linewidth=0)

    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    plt.savefig(file_name, bbox_inches="tight")
    file_name = file_name.parent / f"{file_name.name[:-4]}.png"
    plt.savefig(file_name, dpi=600, bbox_inches="tight")
    plt.close()


def plot_discharge(obs: np.ndarray, preds: np.ndarray, ds: CamelsTXT,
                   figsize: Tuple[float, float], cfg: Dict, basin: str,
                   nse: float):
    fig, ax = plt.subplots(figsize=figsize)

    start_id = 1 + 3*365 # to start with 1.10.xxxx
    start_date = ds.period_start + pd.DateOffset(days=start_id+364)
    end_id = start_id + 2*365
    end_date = ds.period_start + pd.DateOffset(days=end_id + 364)
    dates = pd.date_range(start_date, end_date - pd.DateOffset(days=1))

    # extract precipitation information
    inputs = ds.x[start_id:end_id, -1, :].numpy()
    inputs = ds.local_rescale(inputs, 'inputs')

    # create two arrays, one for rain, one for snow (t_min < 0)
    rain = inputs[:, 0].copy()
    snow = inputs[:, 0].copy()
    tmean = (inputs[:, 2] + inputs[:, 3]) / 2
    rain[tmean <= 0] = 0
    snow[tmean > 0] = 0

    # create twin axis for precipitation plot
    ax2 = ax.twinx()
    ax2.invert_yaxis()
    ax2.set_ylim(max(rain.max(), snow.max())*3, 0)
    ax2.set_ylabel('Precipitation (mm/d)')
    fig_rain = ax2.fill_between(dates, 0, rain, step='pre', color='#80b1d3',
                                zorder=1)
    fig_snow = ax2.fill_between(dates, 0, snow, step='pre', color='black',
                                zorder=1)

    y_ticks = [0, 25, 50, 75]
    ax2.set_yticks(y_ticks)

    fig_obs = ax.plot(dates, obs[start_id:end_id], label="observed", zorder=2)
    fig_sim = ax.plot(dates, preds[start_id:end_id], '--', label="predicted",
                      zorder=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_ylim(0, max(obs[start_id:end_id].max(),
                       preds[start_id:end_id].max())*1.7)
    ax.set_xlim(start_date, end_date)
    title = [f"Simulated discharge of Basin: {basin} - NSE of test period: ",
             f"{nse:.4f}"]
    ax.set_title("".join(title))

    lns = fig_obs + fig_sim + [fig_rain] + [fig_snow]
    labs = ['observed discharge', "simulated discharge", "rain", "snow"]
    ax.legend(lns, labs, loc="center left")#, bbox_to_anchor=(0.1, 0.5))

    file_name = cfg["plot_dir"] / "discharge.svg"
    plt.savefig(file_name, bbox_inches='tight')

    file_name = cfg["plot_dir"] / "discharge.png"
    plt.savefig(file_name, dpi=600, bbox_inches='tight')
    plt.close()


def plot_snow_cell(basin: str, c_ns: np.ndarray, cell: int, sample: int,
                   cfg: Dict, figsize: Tuple[float, float]):
    # get feature-wise mean and std of training set
    ds_train = CamelsTXT(cfg["camels_root"], basin, period="n_cal",
                         years=[0, 20])
    means = ds_train.get_means()
    stds = ds_train.get_stds()

    # load test data to extract inputs
    ds = CamelsTXT(cfg["camels_root"], basin, period="n_test",
                        years=[25, -1], means=means, stds=stds)

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

    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.5
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=figsize, sharex='col',
                           sharey="row")

    offset = 365
    # first column t = sample - offset
    start_date = ds.period_start + pd.DateOffset(days=sample-offset)
    end_date = start_date + pd.DateOffset(days=365 - 1)
    dates = pd.date_range(start_date, end_date)
    inputs_ = ds.local_rescale(ds.x[sample - offset, :, :].numpy(), 'inputs')

    tmean = (inputs_[:, 2] + inputs_[:, 3])*0.5
    rain = inputs_[:, 0].copy()
    snow = rain.copy()
    rain[tmean <= 0] = 0
    snow[tmean > 0] = 0
    ax[0, 0].fill_between(dates, 0, rain, step='pre',
                          facecolor='#80b1d3', label='rain')
    ax[0, 0].fill_between(dates, 0, snow, step='pre',
                          facecolor='black', label='snow')
    ax[0, 0].set_title(f"Day of prediction: {end_date.date()}")
    ax[0, 0].set_ylabel('Precipitation [mmd-1]')
    ax[0, 0].legend(loc="lower right")

    swe_seq = df.loc[start_date:end_date, 'SWE'].values
    swe_seq = MinMaxScaler().fit_transform(swe_seq.reshape(-1,1))
    ax[1, 0].plot(dates, swe_seq, label="Snow water equivalent")
    cell_values = c_ns[sample-offset, :, cell].reshape(-1,1)
    cell_values = MinMaxScaler().fit_transform(cell_values)
    ax[1, 0].plot(dates, cell_values, '--', label='Cell state')
    ax[1, 0].set_ylabel("Normalized values")

    ax[2, 0].plot(dates, inputs_[:, 2], label="daily max", color='#ef8a62')
    ax[2, 0].plot(dates, inputs_[:, 3], label="daily min", color='#67a9cf')
    ax[2, 0].plot([dates[0], dates[-1]], [0, 0], color='lightgray', zorder=0)
    ax[2, 0].set_ylabel("Temperature [°C]")
    ax[2, 0].set_xlim(dates[0], dates[-1])
    ax[2, 0].set_xlabel("Corresponding date to input time step")
    for i, tick in enumerate(ax[2, 0].get_xticklabels()):
        if i % 2 == 0:
            tick.set_rotation(25)
        else:
            tick.set_visible(False)

    # second column t = sample
    start_date = ds.period_start + pd.DateOffset(days=sample)
    end_date = start_date + pd.DateOffset(days=365 - 1)
    dates = pd.date_range(start_date, end_date)
    inputs_ = ds.local_rescale(ds.x[sample, :, :].numpy(), 'inputs')

    tmean = (inputs_[:, 2] + inputs_[:, 3])*0.5
    rain = inputs_[:, 0].copy()
    snow = rain.copy()
    rain[tmean <= 0] = 0
    snow[tmean > 0] = 0
    ax[0, 1].fill_between(dates, 0, rain, step='pre',
                          facecolor='#80b1d3')
    ax[0, 1].fill_between(dates, 0, snow, step='pre',
                          facecolor='black')
    ax[0, 1].set_title(f"Day of prediction: {end_date.date()}")

    swe_seq = df.loc[start_date:end_date, 'SWE'].values
    swe_seq = MinMaxScaler().fit_transform(swe_seq.reshape(-1,1))
    ax[1, 1].plot(dates, swe_seq, label="Snow water equivalent")
    cell_values = c_ns[sample, :, cell].reshape(-1,1)
    cell_values = MinMaxScaler().fit_transform(cell_values)
    ax[1, 1].plot(dates, cell_values, '--', label='Cell state')
    ax[1, 1].legend()

    ax[2, 1].plot(dates, inputs_[:, 2], label="daily max", color='#ef8a62')
    ax[2, 1].plot(dates, inputs_[:, 3], label="daily min", color='#67a9cf')
    ax[2, 1].plot([dates[0], dates[-1]], [0, 0], color='lightgray', zorder=0)
    ax[2, 1].set_xlim(dates[0], dates[-1])
    ax[2, 1].set_xlabel("Corresponding date to input time step")
    for i, tick in enumerate(ax[2, 1].get_xticklabels()):
        if i % 2 == 0:
            tick.set_rotation(25)
        else:
            tick.set_visible(False)

    # first column t = sample + offset
    start_date = ds.period_start + pd.DateOffset(days=sample+offset)
    end_date = start_date + pd.DateOffset(days=365 - 1)
    dates = pd.date_range(start_date, end_date)
    inputs_ = ds.local_rescale(ds.x[sample + offset, :, :].numpy(), 'inputs')

    tmean = (inputs_[:, 2] + inputs_[:, 3])*0.5
    rain = inputs_[:, 0].copy()
    snow = rain.copy()
    rain[tmean <= 0] = 0
    snow[tmean > 0] = 0
    ax[0, 2].fill_between(dates, 0, rain, step='pre',
                          facecolor='#80b1d3')
    ax[0, 2].fill_between(dates, 0, snow, step='pre',
                          facecolor='black')
    ax[0, 2].set_title(f"Day of prediction: {end_date.date()}")
    ax[0, 2].invert_yaxis()
    ylim = ax[0, 2].get_ylim()
    ax[0, 2].set_ylim(ylim[0], 0)
    ax[0, 2].set_title(f"Day of prediction: {end_date.date()}")

    swe_seq = df.loc[start_date:end_date, 'SWE'].values
    swe_seq = MinMaxScaler().fit_transform(swe_seq.reshape(-1,1))
    ax[1, 2].plot(dates, swe_seq, label="Snow water equivalent")
    cell_values = c_ns[sample+offset, :, cell].reshape(-1,1)
    cell_values = MinMaxScaler().fit_transform(cell_values)
    ax[1, 2].plot(dates, cell_values, '--', label='Cell state')

    inputs_ = ds.local_rescale(ds.x[sample+offset, :, :].numpy(), 'inputs')
    ax[2, 2].plot(dates, inputs_[:, 2], label="daily max", color='#ef8a62')
    ax[2, 2].plot(dates, inputs_[:, 3], label="daily min", color='#67a9cf')
    ax[2, 2].plot([dates[0], dates[-1]], [0, 0], color='lightgray', zorder=0)
    ax[2, 2].set_xlim(dates[0], dates[-1])
    ax[2, 2].legend(loc="lower left")
    ax[2, 2].set_xlabel("Corresponding date to input time step")
    ax[2, 2].set_xlim(dates[0], dates[-1])

    for i, tick in enumerate(ax[2, 2].get_xticklabels()):
        if i % 2 == 0:
            tick.set_rotation(25)
        else:
            tick.set_visible(False)

    file_name = cfg["plot_dir"] / 'swe_cell.svg'
    plt.savefig(file_name, bbox_inches='tight')
    plt.tight_layout()
    file_name = cfg["plot_dir"] / 'swe_cell.png'
    plt.savefig(file_name, dpi=600, bbox_inches='tight')

    plt.close()


def plot_scf_cell(basin: str, c_ns: np.ndarray, cell: int, rho:float, p:float,
                  cfg: Dict, figsize: Tuple[float, float]):

    # get feature-wise mean and std of training set
    ds_train = CamelsTXT(cfg["camels_root"], basin, period="n_cal",
                         years=[0, 20])
    means = ds_train.get_means()
    stds = ds_train.get_stds()

    # load test data to extract inputs
    ds = CamelsTXT(cfg["camels_root"], basin, period="n_test",
                   years=[25, -1], means=means, stds=stds)

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

    df = pd.DataFrame(data={f"cell_{cell}": c_ns[:, -1, cell]},
                      index=dates)

    # copy snow cover fraction of basin of interest into cell state data frame
    df["snow_cover"] = df_scf[basin]

    # drop all days without scf measurement
    df = df.dropna(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    scf = df["snow_cover"].values
    scf = StandardScaler().fit_transform(scf.reshape(-1, 1))
    ax.plot(df["snow_cover"].index, scf, label="MODIS SCF")
    cell_values = df[f"cell_{cell}"].values
    cell_values = StandardScaler().fit_transform(cell_values.reshape(-1, 1))
    ax.plot(df["snow_cover"].index, cell_values, label="LSTM cell")
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized scf/cell state")
    ax.set_title(f"Snow cover fraction (SCF) cell: corr={rho:.2f}, p={p:.2e}")
    ax.set_yticklabels([])
    ax.legend()

    file_name = cfg["plot_dir"] / 'scf_cell.svg'
    plt.savefig(file_name, bbox_inches='tight')
    plt.tight_layout()
    file_name = cfg["plot_dir"] / 'scf_cell.png'
    plt.savefig(file_name, dpi=600, bbox_inches='tight')

    plt.close()

def plot_et_cell(basin: str, h_ns: np.ndarray, cell: int, rho: float, p: float,
                 cfg: Dict, figsize=Tuple[float, float]):
    # get feature-wise mean and std of training set
    ds_train = CamelsTXT(cfg["camels_root"], basin, period="n_cal",
                         years=[0, 20])
    means = ds_train.get_means()
    stds = ds_train.get_stds()

    # load test data to extract inputs
    ds = CamelsTXT(cfg["camels_root"], basin, period="n_test",
                   years=[25, -1], means=means, stds=stds)

    # load snow-cover fraction
    scf_file = cfg["data_dir"] / 'ET_basins_MOD16A2.txt'
    df_scf = pd.read_csv(scf_file, sep=" ", header=0, parse_dates=True,
                         index_col="date")
    columns = df_scf.columns
    # add leading 0 to 7-digit codes
    columns = [c.zfill(8) if (len(c) == 7) else c for c in columns]
    df_scf.columns = columns

    # second data frame with last cell state of each sequence
    dates = pd.date_range(start=ds.period_start + pd.DateOffset(days=364),
                          end=ds.period_end)

    df = pd.DataFrame(data={f"cell_{cell}": h_ns[:, -1, cell]},
                      index=dates)

    # copy snow cover fraction of basin of interest into cell state data frame
    df["et"] = df_scf[basin]

    # drop all days without scf measurement
    df = df.dropna(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    scf = df["et"].values
    scf = StandardScaler().fit_transform(scf.reshape(-1, 1))
    ax.plot(df["et"].index, scf, label="MODIS ET")
    cell_values = df[f"cell_{cell}"].values
    cell_values = StandardScaler().fit_transform(cell_values.reshape(-1, 1))
    ax.plot(df["et"].index, cell_values, label="LSTM cell")
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized et/cell output")
    ax.set_title(f"Evapotranspiration (ET) cell: corr={rho:.2f}, p={p:.2e}")
    ax.set_yticklabels([])
    ax.legend()

    file_name = cfg["plot_dir"] / 'et_cell.svg'
    plt.savefig(file_name, bbox_inches='tight')
    plt.tight_layout()
    file_name = cfg["plot_dir"] / 'et_cell.png'
    plt.savefig(file_name, dpi=600, bbox_inches='tight')

    plt.close()


def plot_inputs(basin: str, sample: int, cfg: Dict,
                figsize=Tuple[float, float]):
    ds = CamelsTXT(cfg["camels_root"], basin, period="n_cal",
                   years=[0, 20])
    inputs, output = ds.__getitem__(sample)
    inputs = ds.local_rescale(inputs.numpy(), variable="inputs")
    output = ds.local_rescale(output.numpy(), variable="output")

    fig, ax = plt.subplots(ncols=4, figsize=figsize)

    start_date = ds.period_start + pd.DateOffset(days=sample)
    end_date = start_date + pd.DateOffset(days=365)
    dates = pd.date_range(start_date, end_date - pd.DateOffset(days=1))

    # create two arrays, one for rain, one for snow (t_min < 0)
    rain = inputs[:, 0].copy()
    snow = inputs[:, 0].copy()
    tmean = (inputs[:, 2] + inputs[:, 3]) / 2
    rain[tmean <= 0] = 0
    snow[tmean > 0] = 0

    ax[0].fill_between(dates, 0, rain, step='pre', facecolor='#80b1d3',
                       zorder=1, label="rain")
    ax[0].fill_between(dates, 0, snow, step='pre', facecolor='black',
                       zorder=0, label="snow")
    ax[0].set_ylabel("Precipitation (mm/d)")
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(25)
    ax[0].legend()

    ax[1].plot(dates, inputs[:, 2], label="daily max", color='#ef8a62',
               linewidth=0.5)
    ax[1].plot(dates, inputs[:, 3], label="daily min", color='#67a9cf',
               linewidth=0.5)
    ax[1].set_ylabel("Temperature (°C)")
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(25)
    ax[1].legend()

    ax[2].plot(dates, inputs[:, 1], label="solar radiation", color="#f97306",
               linewidth=0.5)
    ax[2].set_ylabel("Solar radiation (Wm-2)")
    for tick in ax[2].get_xticklabels():
        tick.set_rotation(25)
    ax[2].legend()

    ax[3].plot(dates, inputs[:, 4], label="vapor pressure", color="#bf77f6",
               linewidth=0.5)
    ax[3].set_ylabel("Vapor pressure (Pa)")
    for tick in ax[3].get_xticklabels():
        tick.set_rotation(25)
    ax[3].legend()

    file_name = cfg["plot_dir"] / 'inputs.svg'
    plt.savefig(file_name, bbox_inches='tight')
    plt.tight_layout()
    file_name = cfg["plot_dir"] / 'inputs.png'
    plt.savefig(file_name, dpi=600, bbox_inches='tight')

    plt.close()
