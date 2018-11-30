# Do internals of neural networks make sense in the context of hydrology?

This folder contains all code to reproduce the results presented at the AGU 2018 Fall meeting. It also contains the already trained model weights, as well as the results of the evaluation.

To be able to train the models from scratch, as well as to evaluate the trained models, the user has to download the publicly available [CAMELS data set](https://ral.ucar.edu/solutions/products/camels). The parts needed are "CAMELS time series meteorology, observed flow, meta data (.zip)" and "CAMELS time series Daymet forced model output (.zip)" The content of both archives has to be extracted together into a single folder and this path has to be provided as input argument to the run_experiment.py script.

The structure of this folder is as follows:
- [data](data/) contains MODIS data, CAMELS catchment attributes and a list containing the best random seed for each basin of the SAC-SMA + Snow-17 benchmark model provided in the CAMELS data set.
- [plots](plots/) contains the figures produced from evaluating the trained models
- [results](results/) contains the LSTM weights for each basin, each random seed and each epoch. For each basin, 10 different LSTMs are trained strating with different random seeds. The model with the highest accuracy in the validation period is used for the final evaluation of the test set.


## Requirements

It is recommended to use Anaconda/Miniconda. I report the package versions I used, which don't have to be the most recent. 

- Python >= 3.6
- PyTorch = 0.4.1
- Pandas = 0.23.4
- Numpy = 1.15.2
- Scipy = 1.1.0
- tydm = 4.23.4
- Matplotlib = 2.2.2
- Scikit-Learn = 0.20
- TensorboardX = 1.4
- Numba = 0.40

For training the LSTMs from scratch it is recommended to have a CUDA capable NVIDIA GPU and to have the PyTorch GPU version installed.


## Working with the Python code

The only code that has to be used is the `run_experiments.py` file. This file has to be started with the following arguments:

- `--camels_root`: `str` Path to the folder containing the CAMELS meteorological forcing, streamflow records and SAC-SMA model outputs.
- `--tensorboard`: `bool` (`True`/`False`) If training statistics should be written into tensorboard log files.
- `--mode`: One of [`train`, `eval`, `both`]. Determines whether the models should be trained from scratch (`train`), the trained models should only be evaluate (`eval`) or both should be made (`both`). Note: Training the models (10 models per basin) takes some time, especially on CPU only. Therefore the trained weights are included and you can directly run the evaluation mode.

## Train models from scratch

```
python run_experiments.py --camels_root /path/to/CAMELS/ --mode train --tensorboard True
```

## Evaluate the trained models

Will produce the figures shown in the presentation:
```
python run_experiments.py --camels_root /path/to/CAMELS/ --mode eval
```

## First train, the evaluate

```
python run_experiments.py --camels_root /path/to/CAMELS/ --mode both --tensorboard True
```

## Using Tensorboard during training

If `--tensorboard True` is passed as input argument during trainnig, Tensorboard log files are written in the results folder. You can observed the training progress calling tensorboard from the command line with the following argument.

`> tensorboard --logdir /path/to/results/folder`

For this, you have to have TensorBoard installed on your computer.
