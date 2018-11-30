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
from typing import List

import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

from utils import reshape_data, load_discharge, load_forcing


class CamelsTXT(Dataset):
    """Torch Dataset for basic use of data from the CAMELS data set.

    This data set provides meteorological observations and discharge of a given
    basin from the CAMELS data set.
    """
    def __init__(self, camels_root: PosixPath, basin: str, period: str=None,
                 years: List=None, means: pd.Series=None, stds: pd.Series=None):
        """Initialize Dataset containing the data of a single basin.

        :param camels_root: Full path to the root directory of the CAMELS data
            set.
        :param basin: 8-digit code of basin as string.
        :param period: (optional) One of ['n_cal', 'n_val', 'n_test'.
            n_cal/val/test' needs start and end year specified via 'years'.
        :param means: If any other than calibration period is chosen and
            'global_normalization' is False then means and stds of the each
            feature has to be provided. (Calculated from calibration period.
            Use .get_means() and .get_stds() on training dataset.
        :param stds: see above.
        """
        self.camels_root = camels_root
        self.basin = basin
        self.period = period
        self.years = years

        if self.period != 'n_cal':
            self.means = means
            self.stds = stds
        else:
            # have to be calculated in this dataset
            self.means = None
            self.stds = None

        self.x, self.y = self._load_data()

        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output data from text files."""
        df, area = load_forcing(self.camels_root, self.basin)
        df['QObs(mm/d)'], cal_start = load_discharge(self.camels_root,
                                                     self.basin, area)

        # determine dates of start and end of period
        if self.period == 'n_cal':
            end_year = cal_start.year + self.years[-1]
            cal_end = pd.to_datetime(f"{end_year}/09/30",
                                     yearfirst=True)

        if self.period == 'n_cal':
            self.means = df[cal_start:cal_end].mean()
            self.stds = df[cal_start:cal_end].std()

        # Check which period of the time series is requested
        if self.period == 'n_cal':
            df = df[cal_start:cal_end]

        else:
            start_year = cal_start.year + self.years[0]
            start_date = pd.to_datetime(f"{start_year}/10/01", yearfirst=True)

            # check if end date is till end of time series
            if self.years[1] < 0:
                end_date = df.index[-1]
            else:
                end_year = cal_start.year + self.years[1]
                end_date = pd.to_datetime(f"{end_year}/09/30", yearfirst=True)
            df = df[start_date:end_date]

        # store first and last date of the selected period
        self.period_start = df.index[0]
        self.period_end = df.index[-1]

        # extract data matrix from data frame
        x = np.array([df['prcp(mm/day)'].values,
                      df['srad(W/m2)'].values,
                      df['tmax(C)'].values,
                      df['tmin(C)'].values,
                      df['vp(Pa)'].values]).T

        y = np.array([df['QObs(mm/d)'].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self._local_normalization(x, variable='inputs')
        x, y = reshape_data(x, y, 365)

        if self.period != "n_test":
            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['prcp(mm/day)'],
                              self.means['srad(W/m2)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['vp(Pa)']])
            stds = np.array([self.stds['prcp(mm/day)'],
                             self.stds['srad(W/m2)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['vp(Pa)']])
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["QObs(mm/d)"]) /
                       self.stds["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['prcp(mm/day)'],
                              self.means['srad(W/m2)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['vp(Pa)']])
            stds = np.array([self.stds['prcp(mm/day)'],
                             self.stds['srad(W/m2)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['vp(Pa)']])
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["QObs(mm/d)"] +
                       self.means["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def get_means(self) -> pd.Series:
        """Get feature wise means.

        :return: pd.Series containing feature means.
        """
        return self.means

    def get_stds(self) -> pd.Series:
        """Get feature wise stds.

        :return: pd.Series containing feature stds.
        """
        return self.stds

