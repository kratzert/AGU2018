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

import random
from typing import Dict

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

from camelstxt import CamelsTXT
from model import Model
from utils import calc_nse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(basin: str, seed: int, cfg: Dict):
    """Train LSTM for a given basin.

    :param basin: 8-digit basin code of CAMELS data set
    :param seed: integer specifying the random seed
    :param cfg: Dictionary containing several run options.
    """
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    run_dir = cfg["model_dir"] / basin / f"seed_{seed}"

    # prepare data input pipelines
    ds_train = CamelsTXT(cfg["camels_root"], basin=basin, period='n_cal',
                         years=[0, 20])

    # get feature means and stds of calibration period
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_val = CamelsTXT(cfg["camels_root"], basin=basin, period='n_val',
                       years=[20, 25], means=means, stds=stds)

    # Initialize Dataloader
    tr_loader = DataLoader(ds_train, batch_size=256, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=2048)

    # prepare writer for this run
    if cfg["tensorboard"]:
        writer = SummaryWriter(log_dir=run_dir)

    # initialie model, optimizer, learning rate scheduler
    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Models are trained by minimizing MSE
    loss_func = torch.nn.MSELoss()

    # determine number of total training steps per epoch
    n_samples = len(tr_loader.dataset)
    n_steps = np.ceil(n_samples / tr_loader.batch_size).astype(np.int32)

    global_step = 1

    learning_rates = {10: 2e-3,
                      20: 1e-3,
                      30: 5e-4,
                      35: 1e-4}

    # Start training cycle
    for epoch in range(1, 41):

        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        # Train routine for a single epoch
        model.train()

        for i, (x, y) in enumerate(tr_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            predictions, _, _ = model(x)
            loss = loss_func(predictions, y)
            # calculate gradients
            loss.backward()
            # perform parameter update
            optimizer.step()

            if (i + 1) % 5 == 0:
                print(
                    f"Epoch {epoch} - Step {i+1}/{n_steps}: Loss {loss.item()}")

                if cfg["tensorboard"]:
                    writer.add_scalar('train/loss', loss.item(), global_step)

            global_step += 1

        # Evaluation routine on validation set
        model.eval()
        preds, obs = None, None
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                p, _, _ = model(x)

                if preds is None:
                    preds = p.cpu()
                    obs = y
                else:
                    preds = torch.cat((preds, p.cpu()), 0)
                    obs = torch.cat((obs, y), 0)

            loss = loss_func(preds, obs)

            obs = ds_val.local_rescale(obs.numpy(), variable='output')
            preds = ds_val.local_rescale(preds.numpy(), variable='output')

            # negative discharge predictions are clipped to zero
            preds[preds < 0] = 0

            nse = calc_nse(obs=obs, sim=preds)
            print(f"===> Epoch {epoch} & Basin {basin} NSE: {nse:.5f}")

            if cfg["tensorboard"]:
                writer.add_scalar('val/nse', nse, global_step)

        # save weights
        weight_file = run_dir / f'model_epoch{epoch}_nse{nse}.pt'
        torch.save(model.state_dict(), str(weight_file))