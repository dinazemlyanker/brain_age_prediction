import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn import Dropout, Conv3d, BatchNorm3d, MaxPool3d, AdaptiveAvgPool3d
from typing import Callable, List, Tuple, Union
import numpy as np

class UncertaintyWrapper(pl.LightningModule):
    """ Adding aleatoric uncertainty 
    """

    def __init__(self,
                 mean_model, feature_shape,
                 **kwargs):
        super().__init__()
        self.feature_shape = feature_shape
        self.mean_model = mean_model
        for p in self.mean_model.parameters():
            p.requires_grad = False
        
        self.log_var_head = nn.Linear(feature_shape, 1)
    
    def forward(self, x):
        with torch.no_grad():
            mean, features = self.mean_model(x)
        log_var = self.log_var_head(features)
        return mean, log_var
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        mean, log_var = self.forward(x)
        precision = torch.exp(log_var) + 1e-6
        nll = 0.5 * log_var + 0.5 * ((y - mean) ** 2) / precision
        loss = nll.mean()
        self.log("train_nll", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mean, log_var = self.forward(x)
        precision = torch.exp(log_var) + 1e-6
        nll = 0.5 * log_var + 0.5 * ((y - mean) ** 2) / precision
        loss = nll.mean()
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        return self.forward(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.log_var_head.parameters(), lr=1e-5)
 